"""Hugging Face causal models with a small offline configuration and MPS-safe execution."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config

from open_instruct.portable import data, objectives


def device_for(requested):
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS requested but unavailable. Use native arm64 Python/macOS or explicitly choose --device cpu."
        )
    if requested not in ("cpu", "mps"):
        raise ValueError("Portable backend supports auto, cpu or mps")
    return torch.device(requested)


def create(name, revision, max_length):
    if name == "tiny":
        config = GPT2Config(
            vocab_size=259,
            n_positions=max_length,
            n_embd=64,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        lm = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        tokenizer = data.ByteTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define EOS")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False
        )
        if max_length > lm.config.max_position_embeddings:
            raise ValueError("max_length exceeds model context window")
    for module in lm.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    lm.config.use_cache = False
    return lm, tokenizer


class ValueModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.backbone = lm.base_model
        self.head = nn.Linear(lm.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=1 / (lm.config.hidden_size + 1) ** 0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, attention):
        hidden = self.backbone(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
        return self.head(hidden).squeeze(-1)


def batch(encoded, pad, device):
    length = max(len(tokens) for tokens, _ in encoded)
    tokens = torch.full((len(encoded), length), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    response = torch.zeros((len(encoded), length - 1), dtype=torch.float32, device=device)
    for i, (ids, prompt_length) in enumerate(encoded):
        tokens[i, : len(ids)] = torch.tensor(ids, device=device)
        attention[i, : len(ids)] = 1
        response[i, prompt_length - 1 : len(ids) - 1] = 1
    return tokens, attention, response


def logps(lm, tokens, attention):
    return objectives.token_logps(lm(input_ids=tokens, attention_mask=attention, use_cache=False).logits, tokens)


def sequence_logps(lm, encoded, pad, device, normalize=True):
    tokens, attention, mask = batch(encoded, pad, device)
    scores = (logps(lm, tokens, attention) * mask).sum(-1)
    return scores / mask.sum(-1) if normalize else scores


@torch.no_grad()
def generate(lm, tokenizer, prompt, max_new_tokens, device, sample=False, use_cache=True):
    prefix = data.prompt_tokens(tokenizer, prompt)
    ids = list(prefix)
    past = None
    for _ in range(max_new_tokens):
        tokens = torch.tensor([ids if past is None else ids[-1:]], dtype=torch.long, device=device)
        attention = torch.ones((1, len(ids)), dtype=torch.long, device=device)
        output = lm(input_ids=tokens, attention_mask=attention, past_key_values=past, use_cache=use_cache)
        past = output.past_key_values if use_cache else None
        scores = output.logits[0, -1].float()
        # CPU sampling uses the seeded CPU RNG; unmodified categorical policy at temperature 1.
        token = torch.multinomial(scores.softmax(-1).cpu(), 1).item() if sample else scores.argmax().item()
        ids.append(token)
        if token == tokenizer.eos_token_id:
            break
    return ids, len(prefix), tokenizer.decode(ids[len(prefix) :], skip_special_tokens=True)


def save(lm, tokenizer, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    lm.config.save_pretrained(path)
    torch.save({key: value.detach().cpu() for key, value in lm.state_dict().items()}, path / "model.pt")
    if isinstance(tokenizer, data.ByteTokenizer):
        (path / "byte_tokenizer.json").write_text(json.dumps({"vocab_size": 259}))
    else:
        tokenizer.save_pretrained(path)


def restore(path, device):
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    lm = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    lm.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    lm.eval().to(device)
    tokenizer = (
        data.ByteTokenizer() if (path / "byte_tokenizer.json").exists() else AutoTokenizer.from_pretrained(path)
    )
    return lm, tokenizer


@torch.no_grad()
def generate_batch(lm, tokenizer, prompts, max_new_tokens, device):
    """Left-padded, KV-cached greedy evaluation; return unpadded tokens for each example."""
    prefixes = [data.prompt_tokens(tokenizer, prompt) for prompt in prompts]
    width = max(map(len, prefixes))
    tokens = torch.full((len(prefixes), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    for i, prefix in enumerate(prefixes):
        tokens[i, -len(prefix) :] = torch.tensor(prefix, device=device)
        attention[i, -len(prefix) :] = 1
    generated = [[] for _ in prefixes]
    finished = [False for _ in prefixes]
    past = None
    for _ in range(max_new_tokens):
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        output = lm(
            input_ids=tokens,
            attention_mask=attention,
            position_ids=positions if past is None else positions[:, -1:],
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_ids = output.logits[:, -1].argmax(-1).cpu().tolist()
        for i, token in enumerate(next_ids):
            if not finished[i]:
                generated[i].append(token)
                finished[i] = token == tokenizer.eos_token_id
        if all(finished):
            break
        tokens = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
        attention = torch.cat([attention, torch.ones_like(tokens)], dim=1)
    return [
        (prefix + response, len(prefix), tokenizer.decode(response, skip_special_tokens=True))
        for prefix, response in zip(prefixes, generated)
    ]
