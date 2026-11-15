"""Validate completed real-data runs against their frozen protocol and saved raw predictions."""

import json
import math
from collections import Counter
from pathlib import Path

from scripts.portable.real_experiments import record_sources, sha

from open_instruct.portable import data

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), value
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def check_evaluation(result, rows):
    predictions = result["predictions"]
    assert [p["prompt"] for p in predictions] == [r["prompt"] for r in rows]
    assert result["count"] == len(rows)
    assert result["exact_match"] == sum(p["correct"] for p in predictions) / len(rows)
    for task, metrics in result["per_task"].items():
        task_rows = [p for p in predictions if p["task"] == task]
        assert metrics["count"] == len(task_rows)
        assert metrics["accuracy"] == sum(p["correct"] for p in task_rows) / len(task_rows)
        assert metrics["eos_rate"] == sum(p["eos"] for p in task_rows) / len(task_rows)
    finite(result)


def check_overlap(dataset, bank, holdout):
    def spans(text):
        words = data.normalize(text).split()
        return {tuple(words[i : i + 13]) for i in range(max(0, len(words) - 12))}

    eval_rows = [row for rows in bank.values() for row in rows]
    prompts = {data.normalize(row["prompt"]) for row in eval_rows}
    protected = set().union(*(spans(row["prompt"]) for row in eval_rows))
    train_rows = [row for split in ("sft", "preference", "rlvr") for row in dataset[split]] + holdout
    for row in train_rows:
        assert data.normalize(row["prompt"]) not in prompts
        prompt = row["prompt"]
        # Stronger than the preparer's user-turn normalization: include every earlier assistant/system turn too.
        context = "\n".join(message["content"] for message in prompt) if isinstance(prompt, list) else prompt
        text = context + " " + row.get("chosen", "") + " " + row.get("rejected", "")
        assert not spans(text) & protected, "13-word overlap in full training context/response"
    return dict(
        training_and_preference_holdout_rows=len(train_rows),
        protected_benchmark_questions=len(eval_rows),
        full_context_13gram_matches=0,
    )


def main():
    plan = read(BASE / "plan.json")
    assert record_sources() == plan["sources"], "Source differs from frozen plan"
    assert sha(BASE / "data-v2/manifest.json") == plan["data_manifest_sha256"]
    assert sha(ROOT / "configs/portable/tulu3-real.json") == plan["config_sha256"]
    dataset = data.load(BASE / "data-v2/dataset.json")
    bank = read(BASE / "data-v2/benchmarks.json")
    overlap = check_overlap(dataset, bank, read(BASE / "data-v2/preference_holdout.json"))
    validated = {}
    pending = []
    names = [f"main-seed{s}" for s in plan["seeds"]] + list(plan["branches"])
    for name in names:
        folder = BASE / name
        if not (folder / "summary.json").exists():
            pending.append(name)
            continue
        summary = read(folder / "summary.json")
        assert summary["status"] == "completed"
        assert summary["dataset_sha256"] == data.digest(dataset)
        cfg = summary["config"]
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        finite(metrics)
        counts = Counter(row["stage"] for row in metrics)
        assert counts["rlvr"] == cfg["rlvr_steps"] * cfg["ppo_epochs"]
        rollouts = [json.loads(line) for line in (folder / "rollouts.jsonl").read_text().splitlines()]
        assert len(rollouts) == cfg["rlvr_steps"] * cfg["batch_size"]
        assert all(len(r["tokens"]) <= cfg["max_new_tokens"] for r in rollouts)
        task = summary.get("rlvr_task", "mixed")
        allowed = {data.normalize(r["prompt"]) for r in dataset["rlvr"] if task == "mixed" or r["task"] == task}
        used = [data.normalize(r["prompt"]) for r in rollouts]
        assert set(used) <= allowed
        if cfg["sampling"] == "epochs" and len(rollouts) <= len(allowed):
            assert len(set(used)) == len(rollouts)
        for row in rollouts:
            expected = row["verified"] * cfg["reward_scale"] if row["eos"] else cfg["missing_eos_penalty"]
            assert row["reward"] == expected
        evaluations = read(folder / "evaluations.json")
        if name.startswith("main-seed"):
            assert counts == Counter(
                sft=cfg["sft_steps"],
                reward_model=cfg["rm_steps"],
                dpo=cfg["dpo_steps"],
                rlvr=cfg["rlvr_steps"] * cfg["ppo_epochs"],
            )
            for stage in ("base", "sft", "dpo", "rlvr"):
                for split in ("validation", "test"):
                    check_evaluation(evaluations[stage][split], dataset[split])
            checkpoint_hashes = {stage: sha(folder / "checkpoints" / stage / "model.pt") for stage in evaluations}
            assert len(set(checkpoint_hashes.values())) == 4, "Each stage must update the policy"
            expected_sources = {k: v for k, v in plan["sources"].items() if k.startswith("open_instruct/portable/")}
            assert summary["environment"]["source_sha256"] == expected_sources
        else:
            check_evaluation(evaluations, dataset["test"])
            assert summary["source_sha256"] == plan["sources"]
            assert summary["anchor_sha256"] == sha(BASE / "main-seed1/checkpoints" / summary["anchor"] / "model.pt")
            checkpoint_hashes = {"rlvr": sha(folder / "checkpoints/rlvr/model.pt")}
        validated[name] = dict(
            metric_records=len(metrics),
            rollouts=len(rollouts),
            checkpoints=checkpoint_hashes,
            unique_rl_prompts=len(set(used)),
            allowed_rl_prompts=len(allowed),
        )
    for task, rows in bank.items():
        folder = BASE / "benchmark-rlvr-128"
        path = folder / f"{task}-summary.json"
        if not path.exists():
            pending.append("benchmark/" + task)
            continue
        summary = read(path)
        predictions = [json.loads(line) for line in (folder / f"{task}-predictions.jsonl").read_text().splitlines()]
        assert len(predictions) == len(rows) == summary["count"]
        assert [p["row_sha256"] for p in predictions] == [data.digest(row) for row in rows]
        assert summary["correct"] == sum(p["correct"] for p in predictions)
        assert summary["accuracy"] == summary["correct"] / len(rows)
        assert summary["source_sha256"] == plan["sources"]
        assert summary["checkpoint_sha256"] == sha(BASE / "main-seed1/checkpoints/rlvr/model.pt")
        validated["benchmark/" + task] = dict(predictions=len(rows), correct=summary["correct"])
    path = BASE / "preference-holdout-results.json"
    if path.exists():
        holdout = read(path)
        rows = read(BASE / "data-v2/preference_holdout.json")
        assert holdout["dataset_sha256"] == data.digest(rows)
        assert holdout["source_sha256"] == plan["sources"]
        for name in [f"main-seed{s}" for s in plan["seeds"]]:
            for stage in ("base", "sft", "dpo", "rlvr"):
                result = holdout["runs"][name][stage]
                assert result["count"] == len(rows) == 16
                assert result["checkpoint_sha256"] == sha(BASE / name / "checkpoints" / stage / "model.pt")
                assert [p["prompt"] for p in result["predictions"]] == [r["prompt"] for r in rows]
                assert all(p["correct"] == (p["chosen_logp"] > p["rejected_logp"]) for p in result["predictions"])
                assert result["accuracy"] == sum(p["correct"] for p in result["predictions"]) / 16
        validated["preference-holdout"] = dict(runs=3, stages=4, examples=16)
    else:
        pending.append("preference-holdout")
    for stage in ("base", "rlvr"):
        path = BASE / f"decode-sensitivity/{stage}-summary.json"
        if not path.exists():
            pending.append("decode-sensitivity/" + stage)
            continue
        summary = read(path)
        predictions = [
            json.loads(line) for line in path.with_name(stage + "-predictions.jsonl").read_text().splitlines()
        ]
        assert len(predictions) == len(dataset["test"]) == 72
        assert [p["row_sha256"] for p in predictions] == [data.digest(r) for r in dataset["test"]]
        assert summary["checkpoint_sha256"] == sha(BASE / "main-seed1/checkpoints" / stage / "model.pt")
        validated["decode-sensitivity/" + stage] = dict(predictions=len(predictions))
    path = BASE / "benchmark-analysis.json"
    if path.exists():
        analysis = read(path)
        for name, digest in analysis["inputs_sha256"].items():
            assert sha(BASE / name) == digest
        assert sum(v["count"] for v in analysis["math"]["subject"].values()) == 500
        validated["benchmark-analysis"] = dict(
            math_subjects=len(analysis["math"]["subject"]), instruction_types=len(analysis["ifeval"]["by_instruction"])
        )
    else:
        pending.append("benchmark-analysis")
    result = dict(status="passed" if not pending else "partial", validated=validated, pending=pending, overlap=overlap)
    (BASE / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
