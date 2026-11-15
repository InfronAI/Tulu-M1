"""Bounded single-device stages with cached DPO references and PPO value initialization."""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from open_instruct import logger_utils
from open_instruct.portable import data, model, objectives, verifiers

logger = logger_utils.setup_logger(__name__)


@dataclass
class Config:
    model: str = "tiny"
    revision: str = "main"
    device: str = "auto"
    seed: int = 1
    sft_steps: int = 120
    rm_steps: int = 30
    dpo_steps: int = 40
    rlvr_steps: int = 20
    batch_size: int = 4
    micro_batch_size: int = 2
    max_length: int = 128
    max_new_tokens: int = 5
    learning_rate: float = 0.001
    preference_lr: float = 0.0001
    rlvr_lr: float = 0.00003
    dpo_beta: float = 5.0
    kl_beta: float = 0.05
    reward_scale: float = 10.0
    missing_eos_penalty: float = -10.0
    ppo_epochs: int = 4
    normalize_dpo: bool = True
    skip_dpo: bool = False
    value_from_rm: bool = True
    cpu_threads: int = 4
    eval_batch_size: int = 1
    sampling: str = "replacement"
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.0

    def validate(self):
        for name in (
            "sft_steps",
            "rm_steps",
            "dpo_steps",
            "rlvr_steps",
            "batch_size",
            "micro_batch_size",
            "max_length",
            "max_new_tokens",
            "ppo_epochs",
            "cpu_threads",
            "eval_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.micro_batch_size > self.batch_size or self.batch_size % self.micro_batch_size:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if not 0 < self.max_new_tokens < self.max_length:
            raise ValueError("max_new_tokens must be below max_length")
        for name in ("learning_rate", "preference_lr", "rlvr_lr", "dpo_beta", "reward_scale"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"Invalid {name}")
        if not 0 <= self.kl_beta < float("inf"):
            raise ValueError("kl_beta must be finite and nonnegative")
        if not -float("inf") < self.missing_eos_penalty <= 0:
            raise ValueError("missing_eos_penalty must be finite and nonpositive")
        if self.sampling not in ("replacement", "epochs") or self.lr_schedule not in ("constant", "linear"):
            raise ValueError("Unsupported sampling or learning-rate schedule")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        return self


class Recorder:
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()

    def write(self, stage, step, **metrics):
        record = dict(stage=stage, step=step, elapsed_seconds=time.monotonic() - self.started, **metrics)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if step == 1 or step % 10 == 0:
            logger.info("%s step %s: %s", stage, step, metrics)


def step_optimizer(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    first = optimizer.param_groups[0]["params"][0]
    model.release_cache(first.device)


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sampled_batches(items, cfg):
    pending = []
    while True:
        if cfg.sampling == "replacement":
            yield random.choices(items, k=cfg.batch_size)
        else:
            while len(pending) < cfg.batch_size:
                epoch = list(items)
                random.shuffle(epoch)
                pending.extend(epoch)
            yield pending[: cfg.batch_size]
            pending = pending[cfg.batch_size :]


def learning_rate(optimizer, base, step, steps, cfg):
    warmup = int(steps * cfg.warmup_ratio)
    factor = 1.0
    if cfg.lr_schedule == "linear":
        factor = step / warmup if warmup and step <= warmup else (steps - step + 1) / max(1, steps - warmup)
    for group in optimizer.param_groups:
        group["lr"] = base * factor
    return base * factor


def supervised(lm, tokenizer, rows, cfg, device, recorder):
    encoded = [data.encode(tokenizer, row, "chosen", cfg.max_length) for row in rows]
    optimizer = torch.optim.AdamW(lm.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
    lm.train()
    batches = sampled_batches(encoded, cfg)
    for step in range(1, cfg.sft_steps + 1):
        selected = next(batches)
        lr = learning_rate(optimizer, cfg.learning_rate, step, cfg.sft_steps, cfg)
        total_tokens = sum(len(ids) - p for ids, p in selected)
        total = 0.0
        for micro in chunks(selected, cfg.micro_batch_size):
            tokens, attention, mask = model.batch(micro, tokenizer.pad_token_id, device)
            loss = -(model.logps(lm, tokens, attention) * mask).sum() / total_tokens
            loss.backward()
            total += loss.item()
        step_optimizer(torch.tensor(total), optimizer, lm.parameters())
        recorder.write("sft", step, loss=total, learning_rate=lr)
    lm.eval()


def train_reward(lm, tokenizer, rows, cfg, device, recorder):
    value = model.ValueModel(model.clone_cpu(lm)).to(device).train()
    chosen = [data.encode(tokenizer, row, "chosen", cfg.max_length) for row in rows]
    rejected = [data.encode(tokenizer, row, "rejected", cfg.max_length) for row in rows]
    optimizer = torch.optim.AdamW(value.parameters(), lr=cfg.preference_lr, weight_decay=0.0)
    batches = sampled_batches(list(range(len(rows))), cfg)
    for step in range(1, cfg.rm_steps + 1):
        total = 0.0
        lr = learning_rate(optimizer, cfg.preference_lr, step, cfg.rm_steps, cfg)
        for indices in chunks(next(batches), cfg.micro_batch_size):
            scores = []
            for encoded in (chosen, rejected):
                tokens, attention, _ = model.batch([encoded[i] for i in indices], tokenizer.pad_token_id, device)
                values = value(tokens, attention)
                scores.append(values.gather(1, (attention.sum(-1) - 1)[:, None]).squeeze(1))
            loss = -F.logsigmoid(scores[0] - scores[1]).sum() / cfg.batch_size
            loss.backward()
            total += loss.item()
        step_optimizer(torch.tensor(total), optimizer, value.parameters())
        recorder.write("reward_model", step, loss=total, learning_rate=lr)
    return value


def preference(lm, tokenizer, rows, cfg, device, recorder, output):
    lm.eval()
    chosen = [data.encode(tokenizer, row, "chosen", cfg.max_length) for row in rows]
    rejected = [data.encode(tokenizer, row, "rejected", cfg.max_length) for row in rows]
    # Cache before updating the policy. No second policy resides in unified memory for DPO.
    with torch.no_grad():
        refs = [
            torch.cat(
                [
                    model.sequence_logps(lm, micro, tokenizer.pad_token_id, device, cfg.normalize_dpo).cpu()
                    for micro in chunks(encoded, cfg.micro_batch_size)
                ]
            )
            for encoded in (chosen, rejected)
        ]
    (output / "dpo_reference.json").write_text(
        json.dumps(
            dict(
                dataset_sha256=data.digest(rows),
                normalize=cfg.normalize_dpo,
                chosen=refs[0].tolist(),
                rejected=refs[1].tolist(),
            ),
            indent=2,
        )
    )
    lm.train()
    optimizer = torch.optim.AdamW(lm.parameters(), lr=cfg.preference_lr, weight_decay=0.0)
    batches = sampled_batches(list(range(len(rows))), cfg)
    for step in range(1, cfg.dpo_steps + 1):
        total = 0.0
        lr = learning_rate(optimizer, cfg.preference_lr, step, cfg.dpo_steps, cfg)
        for indices in chunks(next(batches), cfg.micro_batch_size):
            scores = [
                model.sequence_logps(
                    lm, [encoded[i] for i in indices], tokenizer.pad_token_id, device, cfg.normalize_dpo
                )
                for encoded in (chosen, rejected)
            ]
            loss = objectives.dpo_loss(*scores, refs[0][indices].to(device), refs[1][indices].to(device), cfg.dpo_beta)
            loss = loss * len(indices) / cfg.batch_size
            loss.backward()
            total += loss.item()
        step_optimizer(torch.tensor(total), optimizer, lm.parameters())
        recorder.write("dpo", step, loss=total, learning_rate=lr)


def reinforce(lm, value, tokenizer, rows, cfg, device, recorder, output):
    # CPU placement lowers accelerator pressure; unified-memory machines still store all weights.
    reference = model.clone_cpu(lm).requires_grad_(False)
    parameters = list(lm.parameters()) + list(value.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.rlvr_lr, weight_decay=0.0)
    batches = sampled_batches(rows, cfg)
    for step in range(1, cfg.rlvr_steps + 1):
        lm.eval()
        value.eval()
        selected = next(batches)
        learning_rate(optimizer, cfg.rlvr_lr, step, cfg.rlvr_steps, cfg)
        rollouts, verified, terminal = [], [], []
        with torch.no_grad():
            for row in selected:
                ids, prefix_length, text = model.generate(
                    lm, tokenizer, row["prompt"], cfg.max_new_tokens, device, sample=True
                )
                score = float(data.verify(row, text))
                eos = ids[-1] == tokenizer.eos_token_id
                verified.append(score)
                terminal.append(score * cfg.reward_scale if eos else cfg.missing_eos_penalty)
                rollouts.append((ids, prefix_length))
                with (output / "rollouts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                step=step,
                                prompt=row["prompt"],
                                completion=text,
                                tokens=ids[prefix_length:],
                                verified=score,
                                eos=eos,
                                reward=terminal[-1],
                            )
                        )
                        + "\n"
                    )
            tokens, attention, mask = model.batch(rollouts, tokenizer.pad_token_id, device)
            # Evaluate in microbatches, retaining only detached token-sized tensors.
            old = torch.cat(
                [
                    model.logps(lm, tokens[i : i + cfg.micro_batch_size], attention[i : i + cfg.micro_batch_size])
                    for i in range(0, cfg.batch_size, cfg.micro_batch_size)
                ]
            )
            ref = torch.cat(
                [
                    model.logps(
                        reference,
                        tokens[i : i + cfg.micro_batch_size].cpu(),
                        attention[i : i + cfg.micro_batch_size].cpu(),
                    ).to(device)
                    for i in range(0, cfg.batch_size, cfg.micro_batch_size)
                ]
            )
            old_values = torch.cat(
                [
                    value(tokens[i : i + cfg.micro_batch_size], attention[i : i + cfg.micro_batch_size])[:, :-1]
                    for i in range(0, cfg.batch_size, cfg.micro_batch_size)
                ]
            )
            kl = (old - ref) * mask
            rewards = -cfg.kl_beta * kl
            rewards[torch.arange(cfg.batch_size, device=device), attention.sum(-1) - 2] += torch.tensor(
                terminal, device=device
            )
            advantages, returns = objectives.gae(rewards, old_values, mask)
            mean = objectives.masked_mean(advantages, mask)
            variance = objectives.masked_mean((advantages - mean).square(), mask)
            advantages = (advantages - mean) / (variance + 1e-8).sqrt() * mask
        lm.train()
        value.train()
        for epoch in range(cfg.ppo_epochs):
            total = 0.0
            policy_total = 0.0
            value_total = 0.0
            for start in range(0, cfg.batch_size, cfg.micro_batch_size):
                sl = slice(start, start + cfg.micro_batch_size)
                new = model.logps(lm, tokens[sl], attention[sl])
                values = value(tokens[sl], attention[sl])[:, :-1]
                policy_loss, value_loss = objectives.ppo_loss(
                    new, old[sl], values, old_values[sl], advantages[sl], returns[sl], mask[sl]
                )
                weight = mask[sl].sum() / mask.sum()
                loss = (policy_loss + 0.1 * value_loss) * weight
                loss.backward()
                total += loss.item()
                policy_total += (policy_loss * weight).item()
                value_total += (value_loss * weight).item()
            step_optimizer(torch.tensor(total), optimizer, parameters)
            recorder.write(
                "rlvr",
                step,
                epoch=epoch,
                loss=total,
                policy_loss=policy_total,
                value_loss=value_total,
                verified_rate=sum(verified) / len(verified),
                mean_reward=sum(terminal) / len(terminal),
                sampled_kl=objectives.masked_mean(kl, mask).item(),
                response_tokens=mask.sum(-1).mean().item(),
            )
    torch.save({k: v.detach().cpu() for k, v in value.state_dict().items()}, output / "checkpoints" / "value.pt")


@torch.no_grad()
def evaluate(lm, tokenizer, rows, cfg, device):
    lm.eval()
    predictions = []
    for micro in chunks(rows, cfg.eval_batch_size):
        generations = model.generate_batch(lm, tokenizer, [row["prompt"] for row in micro], cfg.max_new_tokens, device)
        for row, (ids, prefix, text) in zip(micro, generations):
            grade = (
                verifiers.grade(row, text)
                if row["task"] in ("gsm8k", "math", "ifeval")
                else dict(correct=data.verify(row, text))
            )
            predictions.append(
                dict(
                    prompt=row["prompt"],
                    answer=row.get("answer"),
                    task=row["task"],
                    completion=text,
                    tokens=len(ids) - prefix,
                    eos=ids[-1] == tokenizer.eos_token_id,
                    **grade,
                )
            )
        logger.info("Evaluation progress: %s/%s", len(predictions), len(rows))
        model.release_cache(device)
    paired = [row for row in rows if "chosen" in row and "rejected" in row]
    preference_accuracy = response_nll = None
    if paired:
        encoded = [data.encode(tokenizer, row, "chosen", cfg.max_length) for row in paired]
        chosen = torch.cat(
            [
                model.sequence_logps(lm, micro, tokenizer.pad_token_id, device)
                for micro in chunks(encoded, cfg.micro_batch_size)
            ]
        )
        rejected = [data.encode(tokenizer, row, "rejected", cfg.max_length) for row in paired]
        rejected_scores = torch.cat(
            [
                model.sequence_logps(lm, micro, tokenizer.pad_token_id, device)
                for micro in chunks(rejected, cfg.micro_batch_size)
            ]
        )
        preference_accuracy = (chosen > rejected_scores).float().mean().item()
        response_nll = -chosen.mean().item()
    per_task = {}
    for task in sorted({p["task"] for p in predictions}):
        task_rows = [p for p in predictions if p["task"] == task]
        metrics = dict(
            accuracy=sum(p["correct"] for p in task_rows) / len(task_rows),
            count=len(task_rows),
            eos_rate=sum(p["eos"] for p in task_rows) / len(task_rows),
        )
        if task == "ifeval":
            for mode in ("strict", "loose"):
                checks = [c for p in task_rows for c in p["instruction_" + mode]]
                metrics["instruction_" + mode] = sum(checks) / len(checks)
            metrics["prompt_loose"] = sum(p["prompt_loose"] for p in task_rows) / len(task_rows)
        per_task[task] = metrics
    return dict(
        exact_match=sum(p["correct"] for p in predictions) / len(predictions),
        preference_accuracy=preference_accuracy,
        response_nll=response_nll,
        paired_count=len(paired),
        count=len(rows),
        per_task=per_task,
        predictions=predictions,
    )


def run(cfg, dataset, output):
    cfg.validate()
    data.validate(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = model.device_for(cfg.device)
    if device.type == "mps":
        torch.mps.manual_seed(cfg.seed)
    lm, tokenizer = model.create(cfg.model, cfg.revision, cfg.max_length)
    for rows in dataset.values():
        for row in rows:
            for field in (f for f in ("chosen", "rejected") if f in row):
                data.encode(tokenizer, row, field, cfg.max_length)
            if len(data.prompt_tokens(tokenizer, row["prompt"])) + cfg.max_new_tokens > cfg.max_length:
                raise ValueError("Prompt plus generation budget exceeds max_length")
    lm.to(device).eval()
    recorder = Recorder(output / "metrics.jsonl")
    evaluations = {}

    def snapshot(stage):
        model.release_cache(device)
        model.save(lm, tokenizer, output / "checkpoints" / stage)
        evaluations[stage] = {
            split: evaluate(lm, tokenizer, dataset[split], cfg, device) for split in ("validation", "test")
        }
        (output / "evaluations.json").write_text(json.dumps(evaluations, indent=2, ensure_ascii=False))
        logger.info("%s test exact match %.3f", stage, evaluations[stage]["test"]["exact_match"])

    snapshot("base")
    supervised(lm, tokenizer, dataset["sft"], cfg, device, recorder)
    snapshot("sft")
    value = train_reward(lm, tokenizer, dataset["preference"], cfg, device, recorder)
    value.cpu()
    model.release_cache(device)
    torch.save(
        {k: v.detach().cpu() for k, v in value.state_dict().items()}, output / "checkpoints" / "reward_model.pt"
    )
    if not cfg.skip_dpo:
        preference(lm, tokenizer, dataset["preference"], cfg, device, recorder, output)
    snapshot("dpo" if not cfg.skip_dpo else "dpo_skipped")
    if not cfg.value_from_rm:
        del value
        value = model.ValueModel(model.clone_cpu(lm))
    value.to(device)
    reinforce(lm, value, tokenizer, dataset["rlvr"], cfg, device, recorder, output)
    snapshot("rlvr")
    return dict(
        device=str(device),
        parameters=sum(p.numel() for p in lm.parameters()),
        resolved_revision=getattr(lm.config, "_commit_hash", None),
        elapsed_seconds=time.monotonic() - recorder.started,
        dataset_sha256=data.digest(dataset),
        evaluations={
            s: {split: {k: v for k, v in metrics.items() if k != "predictions"} for split, metrics in sets.items()}
            for s, sets in evaluations.items()
        },
    )
