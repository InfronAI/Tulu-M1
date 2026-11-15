"""Paired 128-versus-512-token evaluation on the fixed panel; never used to select checkpoints."""

import json
from dataclasses import replace
from pathlib import Path

import torch
from scripts.portable.real_experiments import record_sources, sha

from open_instruct.portable import data, model, train

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"


def main():
    baseline = BASE / "main-seed1"
    summary = json.loads((baseline / "summary.json").read_text())
    cfg = replace(train.Config(**summary["config"]), max_length=8192, max_new_tokens=512, eval_batch_size=4)
    device = model.device_for(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    rows = data.load(BASE / "data-v2/dataset.json")["test"]
    target = BASE / "decode-sensitivity"
    target.mkdir(exist_ok=True)
    for stage in ("base", "rlvr"):
        checkpoint = baseline / "checkpoints" / stage
        receipt = dict(
            stage=stage,
            max_new_tokens=512,
            checkpoint_sha256=sha(checkpoint / "model.pt"),
            dataset_sha256=data.digest(rows),
            source_sha256=record_sources(),
            script_sha256=sha(Path(__file__)),
        )
        header = target / f"{stage}-config.json"
        if header.exists():
            assert json.loads(header.read_text()) == receipt, "Changed sensitivity protocol"
        else:
            header.write_text(json.dumps(receipt, indent=2))
        path = target / f"{stage}-predictions.jsonl"
        predictions = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        assert len(predictions) <= len(rows)
        assert [p["row_sha256"] for p in predictions] == [data.digest(r) for r in rows[: len(predictions)]]
        lm, tokenizer = model.restore(checkpoint, device)
        for start in range(len(predictions), len(rows), cfg.eval_batch_size):
            batch = rows[start : start + cfg.eval_batch_size]
            result = train.evaluate(lm, tokenizer, batch, cfg, device)
            with path.open("a") as stream:
                for row, prediction in zip(batch, result["predictions"]):
                    prediction["row_sha256"] = data.digest(row)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    predictions.append(prediction)
            print(f"Sensitivity {stage}/512: {len(predictions)}/{len(rows)}", flush=True)
        tasks = {}
        short = json.loads((baseline / "evaluations.json").read_text())[stage]["test"]["predictions"]
        for task in ("gsm8k", "math", "ifeval"):
            selected = [p for p in predictions if p["task"] == task]
            original = [p for p in short if p["task"] == task]
            tasks[task] = dict(
                count=len(selected),
                correct=sum(p["correct"] for p in selected),
                short_correct=sum(p["correct"] for p in original),
                eos_rate=sum(p["eos"] for p in selected) / len(selected),
                mean_response_tokens=sum(p["tokens"] for p in selected) / len(selected),
                gained=sum(not a["correct"] and b["correct"] for a, b in zip(original, selected, strict=True)),
                lost=sum(a["correct"] and not b["correct"] for a, b in zip(original, selected, strict=True)),
            )
        (target / f"{stage}-summary.json").write_text(
            json.dumps(dict(**receipt, status="completed", per_task=tasks), indent=2)
        )
        del lm
        model.release_cache(device)


if __name__ == "__main__":
    main()
