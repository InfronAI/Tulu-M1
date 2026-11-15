"""Tulu-data experiments, checkpoint-anchored ablations, and resumable full benchmark coverage."""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from open_instruct.portable import data, model, train
from open_instruct.portable.__main__ import environment

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"
DATA = BASE / "data-v2"
BRANCHES = {
    "control": {},
    "kl-zero": {"kl_beta": 0.0},
    "kl-001": {"kl_beta": 0.01},
    "kl-010": {"kl_beta": 0.1},
    "no-dpo": {"skip_dpo": True},
    "dpo-sum": {"normalize_dpo": False, "dpo_beta": 0.1},
    "policy-critic": {"value_from_rm": False},
    "gsm-only": {"task": "gsm8k"},
    "math-only": {"task": "math"},
    "if-only": {"task": "ifeval"},
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(value)


def record_sources():
    paths = list((ROOT / "open_instruct/portable").glob("*.py"))
    paths += list((ROOT / "open_instruct/IFEvalG").glob("*.py"))
    paths += [ROOT / "open_instruct/math_utils.py", ROOT / "open_instruct/if_functions.py"]
    paths += [ROOT / "scripts/portable/real_experiments.py", ROOT / "scripts/portable/prepare_real_data.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def persist(output, cfg, result):
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed",
                config=asdict(cfg),
                environment=environment(),
                source_sha256=record_sources(),
                **result,
            ),
            indent=2,
        )
    )


def branch(name):
    baseline = BASE / "main-seed1"
    base = read(baseline / "summary.json")
    cfg = train.Config(**base["config"])
    changes = dict(BRANCHES[name])
    task = changes.pop("task", None)
    cfg = replace(cfg, **changes)
    output = BASE / name
    output.mkdir(exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dataset = data.load(DATA / "dataset.json")
    torch.set_num_threads(cfg.cpu_threads)
    device = model.device_for(cfg.device)
    seed(101)
    # Every branch starts from byte-identical saved anchors; optimizer states are fresh.
    anchor = "sft" if cfg.skip_dpo or not cfg.normalize_dpo else "dpo"
    checkpoint = baseline / "checkpoints" / anchor
    lm, tokenizer = model.restore(checkpoint, device)
    value = model.ValueModel(copy.deepcopy(lm)).to(device)
    if cfg.value_from_rm:
        value.load_state_dict(
            torch.load(baseline / "checkpoints/reward_model.pt", weights_only=True, map_location=device)
        )
    recorder = train.Recorder(output / "metrics.jsonl")
    if not cfg.normalize_dpo:
        seed(102)
        train.preference(lm, tokenizer, dataset["preference"], cfg, device, recorder, output)
        model.save(lm, tokenizer, output / "checkpoints/dpo-sum")
    seed(103)  # Common RNG start across every RL branch, including a repeat control.
    rows = [row for row in dataset["rlvr"] if row["task"] == task] if task else dataset["rlvr"]
    train.reinforce(lm, value, tokenizer, rows, cfg, device, recorder, output)
    model.save(lm, tokenizer, output / "checkpoints/rlvr")
    evaluation = train.evaluate(lm, tokenizer, dataset["test"], cfg, device)
    (output / "evaluations.json").write_text(json.dumps(evaluation, indent=2))
    persist(
        output,
        cfg,
        dict(
            kind="anchored_ablation",
            name=name,
            anchor=anchor,
            anchor_sha256=sha(checkpoint / "model.pt"),
            reward_model_sha256=sha(baseline / "checkpoints/reward_model.pt"),
            dataset_sha256=data.digest(dataset),
            rlvr_task=task or "mixed",
            rlvr_examples=len(rows),
            elapsed_seconds=time.monotonic() - recorder.started,
            evaluation={k: v for k, v in evaluation.items() if k != "predictions"},
        ),
    )


def benchmark(stage, max_new_tokens, only_task):
    baseline = BASE / "main-seed1"
    cfg = train.Config(**read(baseline / "summary.json")["config"])
    cfg = replace(cfg, max_new_tokens=max_new_tokens, max_length=8192, eval_batch_size=8)
    device = model.device_for(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    checkpoint = baseline / "checkpoints" / stage
    lm, tokenizer = model.restore(checkpoint, device)
    benchmarks = read(DATA / "benchmarks.json")
    target = BASE / f"benchmark-{stage}-{max_new_tokens}"
    target.mkdir(exist_ok=True)
    for task, rows in benchmarks.items():
        if only_task and task != only_task:
            continue
        receipt = dict(
            stage=stage,
            max_new_tokens=max_new_tokens,
            checkpoint_sha256=sha(checkpoint / "model.pt"),
            dataset_sha256=data.digest(rows),
            count=len(rows),
            source_sha256=record_sources(),
        )
        header = target / f"{task}-config.json"
        if header.exists():
            if read(header) != receipt:
                raise ValueError("Cannot resume benchmark with changed checkpoint/data/protocol/source")
        else:
            header.write_text(json.dumps(receipt, indent=2))
        predictions_path = target / f"{task}-predictions.jsonl"
        predictions = (
            [json.loads(line) for line in predictions_path.read_text().splitlines()]
            if predictions_path.exists()
            else []
        )
        if len(predictions) > len(rows):
            raise ValueError("Too many stored predictions")
        for i, prediction in enumerate(predictions):
            if prediction["row_sha256"] != data.digest(rows[i]):
                raise ValueError("Stored prediction order/data mismatch")
        started = time.monotonic()
        for start in range(len(predictions), len(rows), cfg.eval_batch_size):
            batch = rows[start : start + cfg.eval_batch_size]
            for row in batch:
                if len(data.prompt_tokens(tokenizer, row["prompt"])) + max_new_tokens > cfg.max_length:
                    raise ValueError("Benchmark context overflow; no silently dropped examples")
            result = train.evaluate(lm, tokenizer, batch, cfg, device)
            with predictions_path.open("a") as stream:
                for row, prediction in zip(batch, result["predictions"]):
                    prediction["row_sha256"] = data.digest(row)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    predictions.append(prediction)
            print(f"{stage}/{task}: {len(predictions)}/{len(rows)}", flush=True)
        strict = [c for p in predictions for c in p.get("instruction_strict", [])]
        loose = [c for p in predictions for c in p.get("instruction_loose", [])]
        result = dict(
            **receipt,
            status="completed",
            correct=sum(p["correct"] for p in predictions),
            accuracy=sum(p["correct"] for p in predictions) / len(predictions),
            eos_rate=sum(p["eos"] for p in predictions) / len(predictions),
            response_tokens=sum(p["tokens"] for p in predictions),
            last_invocation_seconds=time.monotonic() - started,
            symbolic_timeouts=sum(p.get("symbolic_timeouts", 0) for p in predictions),
            symbolic_length_skips=sum(p.get("symbolic_length_skips", 0) for p in predictions),
        )
        if strict:
            result.update(
                instruction_strict=sum(strict) / len(strict),
                instruction_loose=sum(loose) / len(loose),
                prompt_loose=sum(p["prompt_loose"] for p in predictions) / len(predictions),
            )
        (target / f"{task}-summary.json").write_text(json.dumps(result, indent=2))


def suite():
    plan = dict(
        seeds=[1, 2, 3],
        branches=BRANCHES,
        full_benchmark=dict(stage="rlvr", max_new_tokens=128, tasks=["gsm8k", "math", "ifeval"]),
        data_manifest_sha256=sha(DATA / "manifest.json"),
        sources=record_sources(),
    )
    plan_path = BASE / "plan.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Frozen experiment plan differs from current source/data")
    plan_path.write_text(json.dumps(plan, indent=2))
    commands = []
    for value in (1, 2, 3):
        name = f"main-seed{value}"
        commands.append(
            (
                name,
                [
                    sys.executable,
                    "-m",
                    "open_instruct.portable",
                    "run",
                    "--config",
                    "configs/portable/tulu3-real.json",
                    "--data",
                    str(DATA / "dataset.json"),
                    "--output",
                    str(BASE / name),
                    "--seed",
                    str(value),
                ],
            )
        )
    for name in BRANCHES:
        commands.append((name, [sys.executable, "-m", "scripts.portable.real_experiments", "branch", "--name", name]))
    commands.append(
        (
            "full-benchmarks",
            [sys.executable, "-m", "scripts.portable.real_experiments", "benchmark", "--stage", "rlvr"],
        )
    )
    for name, command in commands:
        if (BASE / name / "summary.json").exists():
            print(f"Completed already: {name}", flush=True)
            continue
        print(f"Starting {name}", flush=True)
        with (BASE / f"{name}-console.txt").open("a") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("suite")
    run = commands.add_parser("branch")
    run.add_argument("--name", choices=BRANCHES, required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--stage", choices=("base", "sft", "dpo", "rlvr"), default="rlvr")
    bench.add_argument("--max-new-tokens", type=int, default=128)
    bench.add_argument("--task", choices=("gsm8k", "math", "ifeval"))
    args = parser.parse_args()
    if args.command == "suite":
        suite()
    elif args.command == "branch":
        branch(args.name)
    else:
        benchmark(args.stage, args.max_new_tokens, args.task)


if __name__ == "__main__":
    main()
