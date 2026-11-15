"""Evaluate untouched published preference pairs without an external judge or fabricated labels."""

import json
from pathlib import Path

import torch
from scripts.portable.real_experiments import record_sources, sha

from open_instruct.portable import data, model
from open_instruct.portable.__main__ import environment

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"


def main():
    torch.set_num_threads(4)
    rows = json.loads((BASE / "data-v2/preference_holdout.json").read_text())
    results = {}
    for folder in sorted(BASE.glob("main-seed*")):
        if not (folder / "summary.json").exists():
            continue
        stages = {}
        for stage in ("base", "sft", "dpo", "rlvr"):
            lm, tokenizer = model.restore(folder / "checkpoints" / stage, "cpu")
            predictions = []
            with torch.no_grad():
                for row in rows:
                    scores = [
                        model.sequence_logps(
                            lm, [data.encode(tokenizer, row, field, 768)], tokenizer.pad_token_id, "cpu"
                        ).item()
                        for field in ("chosen", "rejected")
                    ]
                    predictions.append(
                        dict(
                            prompt=row["prompt"],
                            chosen_logp=scores[0],
                            rejected_logp=scores[1],
                            correct=scores[0] > scores[1],
                        )
                    )
            stages[stage] = dict(
                count=len(rows),
                accuracy=sum(p["correct"] for p in predictions) / len(rows),
                predictions=predictions,
                checkpoint_sha256=sha(folder / "checkpoints" / stage / "model.pt"),
            )
            del lm
        results[folder.name] = stages
    (BASE / "preference-holdout-results.json").write_text(
        json.dumps(
            dict(
                dataset_sha256=data.digest(rows),
                metric="length-normalized chosen likelihood > rejected likelihood",
                source_sha256=record_sources(),
                script_sha256=sha(Path(__file__)),
                environment=environment(),
                device="cpu",
                runs=results,
            ),
            indent=2,
        )
    )
    print({run: {stage: m["accuracy"] for stage, m in stages.items()} for run, stages in results.items()})


if __name__ == "__main__":
    main()
