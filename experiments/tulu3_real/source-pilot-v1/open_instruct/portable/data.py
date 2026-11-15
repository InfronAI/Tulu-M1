"""Small offline fixtures and strict, inspectable local dataset contracts."""

import hashlib
import json
import random
import re
from pathlib import Path

from open_instruct.portable import verifiers

SPLITS = ("sft", "preference", "rlvr", "validation", "test")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fixture():
    """Synthetic arithmetic/copy tasks. Not GSM8K, MATH or IFEval."""
    rows = []
    for a in range(20):
        for b in range(10):
            rows.append(
                dict(
                    prompt=f"Add {a} and {b}. Answer with only the number.",
                    chosen=str(a + b),
                    rejected=str(a + b + 1),
                    answer=str(a + b),
                    task="exact",
                )
            )
    for a in range(200):
        rows.append(
            dict(
                prompt=f"Copy {a}. Answer with only the number.",
                chosen=str(a),
                rejected=str(a + 1),
                answer=str(a),
                task="exact",
            )
        )
    random.Random(42).shuffle(rows)
    bounds = (0, 200, 280, 336, 368, 400)
    return {split: rows[bounds[i] : bounds[i + 1]] for i, split in enumerate(SPLITS)}


def normalize(text):
    if isinstance(text, list):
        text = "\n".join(message["content"] for message in text if message["role"] == "user")
    return " ".join(text.casefold().split())


def validate(data):
    if set(data) != set(SPLITS):
        raise ValueError(f"Dataset must contain exactly {SPLITS}")
    seen = {}
    for split, rows in data.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty or invalid split: {split}")
        for row in rows:
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                if not prompt or prompt[-1].get("role") != "user":
                    raise ValueError("Chat context must end with a user turn")
                if any(
                    m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str)
                    or not m["content"].strip()
                    for m in prompt
                ):
                    raise ValueError("Invalid chat message")
            elif not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{split}: missing/non-string/empty prompt")
            required = (
                ("chosen",) if split == "sft" else ("chosen", "rejected") if split == "preference" else ("task",)
            )
            for field in set(required) | ({"chosen", "rejected", "answer", "task"} & row.keys()):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{split}: missing/non-string/empty {field}")
            if "rejected" in row and row.get("chosen") == row["rejected"]:
                raise ValueError("Preference completions must differ")
            task = row.get("task")
            if task is not None:
                if task not in ("exact", "integer", "gsm8k", "math", "ifeval"):
                    raise ValueError("Unsupported verifier")
                if task == "ifeval":
                    c = row.get("constraints", {})
                    if "func_name" in c:
                        if c["func_name"] not in verifiers.if_functions.IF_FUNCTIONS_MAP:
                            raise ValueError("Unknown legacy IFEval function")
                    else:
                        if not c.get("instruction_id") or len(c["instruction_id"]) != len(c.get("kwargs", [])):
                            raise ValueError("IFEval needs aligned instruction_id and kwargs lists")
                        unknown = set(c["instruction_id"]) - verifiers.instructions_registry.INSTRUCTION_DICT.keys()
                        if unknown:
                            raise ValueError(f"Unknown IFEval constraints: {unknown}")
                elif not isinstance(row.get("answer"), str) or not row["answer"].strip():
                    raise ValueError("Verifier requires a nonempty answer")
                # Legacy synthetic pairs have deterministic correctness labels. Real preferences do not.
                if (
                    task in ("exact", "integer")
                    and "chosen" in row
                    and "rejected" in row
                    and (not verify(row, row["chosen"]) or verify(row, row["rejected"]))
                ):
                    raise ValueError("Chosen/rejected completions must agree with the deterministic verifier")
            key = normalize(row["prompt"])
            if key in seen:
                raise ValueError(f"Duplicate/contaminated prompt in {seen[key]} and {split}")
            seen[key] = split
    return data


def load(path=None):
    return validate(json.loads(Path(path).read_text()) if path else fixture())


def verify(row, completion):
    if row["task"] == "exact":
        return completion.strip() == row["answer"].strip()
    if row["task"] in ("gsm8k", "math", "ifeval"):
        return verifiers.grade(row, completion)["correct"]
    # Explicit final integer only; incidental numbers in reasoning do not earn reward.
    match = re.search(r"(?:####|Answer:)\s*(-?\d+)\s*$", completion)
    return bool(match and match[1] == row["answer"].strip())


class ByteTokenizer:
    """Reversible UTF-8 codec for offline architecture tests; no external vocabulary."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(t - 3 for t in tokens if t >= 3).decode("utf-8", errors="replace")


def prompt_tokens(tokenizer, prompt):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return [tokenizer.bos_token_id or tokenizer.eos_token_id] + tokenizer.encode(
        "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n", add_special_tokens=False
    )


def encode(tokenizer, row, field, max_length):
    prompt = prompt_tokens(tokenizer, row["prompt"])
    response = tokenizer.encode(row[field], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(response) > max_length:
        raise ValueError(
            f"Sequence exceeds max_length={max_length}; increase it or shorten data. No silent truncation."
        )
    return prompt + response, len(prompt)
