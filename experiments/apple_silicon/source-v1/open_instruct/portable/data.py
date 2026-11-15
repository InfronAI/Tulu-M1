"""Small offline fixtures and strict, inspectable local dataset contracts."""

import hashlib
import json
import random
import re
from pathlib import Path

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
    return " ".join(text.casefold().split())


def validate(data):
    if set(data) != set(SPLITS):
        raise ValueError(f"Dataset must contain exactly {SPLITS}")
    seen = {}
    for split, rows in data.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty or invalid split: {split}")
        for row in rows:
            for field in ("prompt", "chosen", "rejected", "answer", "task"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{split}: missing/non-string/empty {field}")
            if row["chosen"] == row["rejected"]:
                raise ValueError("Preference completions must differ")
            if row["task"] not in ("exact", "integer"):
                raise ValueError("Supported verifiers: exact, integer (not symbolic MATH or full IFEval)")
            if not verify(row, row["chosen"]) or verify(row, row["rejected"]):
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
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
    return [tokenizer.bos_token_id or tokenizer.eos_token_id] + tokenizer.encode(
        f"<|user|>\n{prompt}\n<|assistant|>\n", add_special_tokens=False
    )


def encode(tokenizer, row, field, max_length):
    prompt = prompt_tokens(tokenizer, row["prompt"])
    response = tokenizer.encode(row[field], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(response) > max_length:
        raise ValueError(
            f"Sequence exceeds max_length={max_length}; increase it or shorten data. No silent truncation."
        )
    return prompt + response, len(prompt)
