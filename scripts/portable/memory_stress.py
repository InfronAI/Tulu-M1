"""Exercise longest selected preference examples and a complete PPO update before a long suite."""

import argparse
import json
import random
import resource
from dataclasses import asdict, replace
from pathlib import Path

import torch
from scripts.portable.real_experiments import record_sources

from open_instruct.portable import data, model, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    (args.output / "checkpoints").mkdir()
    cfg = replace(
        train.Config(**json.loads(Path("configs/portable/tulu3-real.json").read_text())),
        sft_steps=1,
        rm_steps=2,
        dpo_steps=2,
        rlvr_steps=1,
    )
    torch.set_num_threads(4)
    torch.manual_seed(9)
    random.seed(9)
    device = model.device_for("mps")
    lm, tokenizer = model.restore(args.checkpoint, device)
    dataset = data.load("experiments/tulu3_real/data-v2/dataset.json")
    pairs = sorted(
        dataset["preference"],
        key=lambda row: sum(
            len(data.encode(tokenizer, row, field, cfg.max_length)[0]) for field in ("chosen", "rejected")
        ),
        reverse=True,
    )[:2]
    rl = sorted(dataset["rlvr"], key=lambda row: len(data.prompt_tokens(tokenizer, row["prompt"])), reverse=True)[:2]
    recorder = train.Recorder(args.output / "metrics.jsonl")
    observations = {}

    def measure(stage):
        model.release_cache(device)
        observations[stage] = dict(
            allocated=torch.mps.current_allocated_memory(),
            driver=torch.mps.driver_allocated_memory(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        print(stage, observations[stage], flush=True)

    value = train.train_reward(lm, tokenizer, pairs, cfg, device, recorder)
    value.cpu()
    measure("reward_model")
    train.preference(lm, tokenizer, pairs, cfg, device, recorder, args.output)
    measure("dpo")
    value.to(device)
    train.reinforce(lm, value, tokenizer, rl, cfg, device, recorder, args.output)
    measure("rlvr")
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                status="completed",
                config=asdict(cfg),
                source_sha256=record_sources(),
                preference_prompts=[r["prompt"] for r in pairs],
                observations=observations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
