"""Check recorded experiment integrity without loading or executing model weights."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments/apple_silicon"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    archive = EXPERIMENTS / "source-v1"
    source = {str(p.relative_to(archive)): sha(p) for p in sorted((archive / "open_instruct/portable").glob("*.py"))}
    evidence = {}
    for summary_path in sorted(EXPERIMENTS.glob("*/summary.json")):
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text())
        cfg = summary["config"]
        dataset = json.loads((folder / "dataset.json").read_text())
        dataset_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert dataset_hash == summary["dataset_sha256"], folder
        assert summary["status"] == "completed", folder
        if folder.name != "mps-seed1":  # Retained pilot predates source-hash recording.
            assert source == summary["environment"]["source_sha256"], folder
        records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        expected = dict(
            sft=cfg["sft_steps"],
            reward_model=cfg["rm_steps"],
            dpo=0 if cfg["skip_dpo"] else cfg["dpo_steps"],
            rlvr=cfg["rlvr_steps"] * cfg["ppo_epochs"],
        )
        for stage, count in expected.items():
            assert sum(r["stage"] == stage for r in records) == count, (folder, stage)
        assert all(math.isfinite(r["loss"]) for r in records), folder
        evaluations = json.loads((folder / "evaluations.json").read_text())
        for stage, splits in evaluations.items():
            for split, metrics in splits.items():
                predictions = metrics["predictions"]
                assert len(predictions) == len(dataset[split]) == metrics["count"]
                assert sum(p["correct"] for p in predictions) / len(predictions) == metrics["exact_match"]
                assert metrics["exact_match"] == summary["evaluations"][stage][split]["exact_match"]
        checkpoints = {
            str(p.relative_to(folder)): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted((folder / "checkpoints").rglob("*.pt"))
        }
        anchor = "dpo_skipped" if cfg["skip_dpo"] else "dpo"
        assert (
            checkpoints[f"checkpoints/{anchor}/model.pt"]["sha256"]
            != checkpoints["checkpoints/rlvr/model.pt"]["sha256"]
        )
        evidence[folder.name] = dict(
            status="verified",
            metric_records=len(records),
            checkpoints=checkpoints,
            source_hashes_match=folder.name != "mps-seed1",
        )
    assert len(evidence) == 10, f"Expected pilot + 9 main runs; found {len(evidence)}"
    (EXPERIMENTS / "artifact-verification.json").write_text(json.dumps(evidence, indent=2))
    print(
        f"Verified {len(evidence)} completed runs, metrics, data hashes, predictions, source identities and checkpoint hashes."
    )


if __name__ == "__main__":
    main()
