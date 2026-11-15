"""Build a bounded, provenance-pinned sample of the released Tulu 3 data and real benchmarks."""

import argparse
import ast
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from huggingface_hub import hf_hub_download
from pyarrow import parquet
from transformers import AutoTokenizer

from open_instruct.portable import data

SOURCES = {
    "sft": (
        "allenai/tulu-3-sft-mixture",
        "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        "data/train-00000-of-00006.parquet",
    ),
    "preference": (
        "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        "78a6f00785946cd24276c5dd075f83a143a3b1e6",
        "data/train-00000-of-00004.parquet",
    ),
    "rlvr": (
        "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints",
        "7dbd180f5440c0b90f2944e6efea934b85437a95",
        "data/train-00000-of-00001.parquet",
    ),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
    "math": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl"),
}
MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def read_source(name, output):
    repo, revision, filename = SOURCES[name]
    filenames = [filename]
    if name in ("sft", "preference"):
        count = 6 if name == "sft" else 4
        filenames = [f"data/train-{i:05d}-of-{count:05d}.parquet" for i in range(count)]
    card = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)
    (output / f"{name}-dataset-card.md").write_text(Path(card).read_text())
    receipts, rows = [], []
    for filename in filenames:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = dict(filename=filename, bytes=path.stat().st_size, sha256=digest)
        if filename.endswith(".jsonl"):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        else:
            source = parquet.ParquetFile(path)
            if name in ("sft", "preference"):
                groups = sorted({int(i * (source.num_row_groups - 1) / 7) for i in range(8)})
                for group in groups:
                    offset = sum(source.metadata.row_group(i).num_rows for i in range(group))
                    for i, row in enumerate(source.read_row_group(group).slice(0, 256).to_pylist()):
                        rows.append(dict(row, _source_row=offset + i, _source_file=filename))
                receipt["row_groups_scanned"] = groups
            else:
                rows.extend(source.read().to_pylist())
        receipts.append(receipt)
    random.Random(42).shuffle(rows)
    if name in ("sft", "preference"):
        # Balanced round-robin over observed sources. This deliberately is not the published mixture ratio.
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["source"]].append(row)
        rows = []
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:
                    rows.append(buckets[key].pop())
    return rows, dict(repo=repo, revision=revision, files=receipts, rows_scanned=len(rows))


def convert(name, row):
    provenance = dict(
        dataset=SOURCES[name][0],
        source_id=str(row.get("id", row.get("key", row.get("_source_row", "")))),
        source=row.get("source", row.get("dataset", name)),
    )
    if name == "sft":
        messages = row["messages"]
        if len(messages) < 2 or messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
            raise ValueError("Unsupported conversation ending")
        return dict(prompt=messages[:-1], chosen=messages[-1]["content"], provenance=provenance)
    if name == "preference":
        chosen, rejected = row["chosen"], row["rejected"]
        if chosen[:-1] != rejected[:-1] or chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
            raise ValueError("Mismatched preference context")
        return dict(
            prompt=chosen[:-1], chosen=chosen[-1]["content"], rejected=rejected[-1]["content"], provenance=provenance
        )
    if name == "rlvr":
        task = row["dataset"].lower()
        prompt = row["messages"][0]["content"]
        original_prompt = prompt
        if task in ("gsm8k", "math"):
            # Explicit zero-shot resource adaptation: discard demonstrations, never the target question.
            if "\n\nQuestion: " not in prompt:
                raise ValueError("Unrecognized few-shot template")
            prompt = prompt.rsplit("\n\nQuestion: ", 1)[-1]
            if prompt.endswith("\nAnswer:"):
                prompt = prompt[:-8]
        result = dict(
            prompt=prompt,
            task=task,
            provenance=provenance,
            original_prompt_sha256=hashlib.sha256(original_prompt.encode()).hexdigest(),
        )
        if task == "ifeval":
            constraints = (
                json.loads(row["ground_truth"])
                if row["ground_truth"].lstrip().startswith("{")
                else ast.literal_eval(row["ground_truth"])
            )
            if isinstance(constraints, list):
                constraints = constraints[0]
            if isinstance(constraints, str):
                constraints = json.loads(constraints)
            result["constraints"] = constraints
        else:
            result["answer"] = row["ground_truth"]
        return result
    if name == "gsm8k":
        return dict(
            prompt=row["question"], answer=row["answer"].split("####")[-1].strip(), task=name, provenance=provenance
        )
    if name == "math":
        return dict(prompt=row["problem"], answer=row["answer"], task=name, provenance=provenance)
    return dict(
        prompt=row["prompt"],
        task="ifeval",
        constraints=dict(instruction_id=row["instruction_id_list"], kwargs=row["kwargs"]),
        provenance=provenance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/tulu3_real/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    nltk.download("punkt_tab", raise_on_error=True, quiet=True)
    nltk.download("punkt", raise_on_error=True, quiet=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    dataset = {split: [] for split in data.SPLITS}
    receipts, benchmark, holdout, rejected, selected_raw = {}, {}, [], Counter(), {}
    seen = set()
    eval_ngrams = set()

    def ngrams(text):
        words = data.normalize(text).split()
        return {tuple(words[i : i + 13]) for i in range(max(0, len(words) - 12))}

    def eligible(row, stage):
        # Bounds selected before training. Never filter based on model correctness.
        if len(data.prompt_tokens(tokenizer, row["prompt"])) + 128 > 768:
            rejected[stage + ":prompt_length"] += 1
            return False
        for field in ("chosen", "rejected"):
            if field in row:
                if not row[field].strip():
                    rejected[stage + ":empty_completion"] += 1
                    return False
                try:
                    data.encode(tokenizer, row, field, 768)
                except ValueError:
                    rejected[stage + ":sequence_length"] += 1
                    return False
        if "chosen" in row and row.get("rejected") == row["chosen"]:
            rejected[stage + ":identical_preference_pair"] += 1
            return False
        if stage in ("sft", "preference", "rlvr"):
            text = data.normalize(row["prompt"]) + " " + row.get("chosen", "") + " " + row.get("rejected", "")
            if ngrams(text) & eval_ngrams:
                rejected[stage + ":13gram_overlap"] += 1
                return False
        key = data.normalize(row["prompt"])
        if not key or key in seen:
            rejected[stage + ":duplicate"] += 1
            return False
        seen.add(key)
        return True

    # Reserve official evaluation prompts before selecting any training examples.
    for name in ("gsm8k", "math", "ifeval"):
        rows, receipts[name] = read_source(name, args.output)
        benchmark[name] = [convert(name, row) for row in rows]
        accepted = []
        for row in benchmark[name]:
            if eligible(row, name):
                accepted.append(row)
                if len(accepted) == 32:
                    break
        dataset["validation"] += accepted[:8]
        dataset["test"] += accepted[8:32]
        # Entire benchmark questions are protected from training even if not in the small stage panel.
        seen.update(data.normalize(row["prompt"]) for row in benchmark[name])
        for row in benchmark[name]:
            eval_ngrams.update(ngrams(row["prompt"]))
    for name, target in (("sft", 96), ("preference", 64), ("rlvr", 72)):
        rows, receipts[name] = read_source(name, args.output)
        selected_raw[name] = []
        counts = Counter()
        for raw in rows:
            try:
                row = convert(name, raw)
            except (KeyError, ValueError, IndexError) as error:
                rejected[name + ":" + str(error)] += 1
                continue
            if name == "rlvr" and counts[row["task"]] >= 24:
                continue
            if not eligible(row, name):
                continue
            if name == "preference" and len(dataset[name]) >= 48:
                holdout.append(row)
            else:
                dataset[name].append(row)
            selected_raw[name].append(raw)
            counts[row.get("task", "all")] += 1
            if len(selected_raw[name]) == target:
                break
        if len(selected_raw[name]) != target:
            raise RuntimeError(
                f"Insufficient eligible {name} rows: {len(selected_raw[name])}/{target}; inspect source and budgets"
            )
    data.validate(dataset)
    (args.output / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    (args.output / "benchmarks.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2))
    (args.output / "preference_holdout.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2))
    (args.output / "selected_original_rows.json").write_text(json.dumps(selected_raw, ensure_ascii=False, indent=2))
    receipt = dict(
        sources=receipts,
        dataset_sha256=data.digest(dataset),
        split_sizes={k: len(v) for k, v in dataset.items()},
        benchmark_sizes={k: len(v) for k, v in benchmark.items()},
        preference_holdout=len(holdout),
        rejected=dict(rejected),
        sft_sources=dict(Counter(r["provenance"]["source"] for r in dataset["sft"])),
        preference_sources=dict(Counter(r["provenance"]["source"] for r in dataset["preference"])),
        rlvr_tasks=dict(Counter(r["task"] for r in dataset["rlvr"])),
        adaptations=[
            "all pinned shards for SFT/preferences; at most 8 row groups x 256 rows per shard; balanced source sampling",
            "last assistant turn supervised; earlier conversation retained as context",
            "normalized prompt and 13-word overlap screening against all 2360 benchmark prompts",
            "length-filtered at 768 tokens; sampling seed 42; no correctness-based filtering",
            "RL math/GSM demonstrations removed; original target and labels preserved",
            "zero-shot task evaluation; 128 response-token cap; not OLMES protocol parity",
        ],
    )
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
