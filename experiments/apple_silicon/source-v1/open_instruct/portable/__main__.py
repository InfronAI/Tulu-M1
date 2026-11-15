"""Run: python -m open_instruct.portable --help."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch

from open_instruct.portable import data, model, train


def environment():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        chip=subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if sys.platform == "darwin"
        else platform.processor(),
        mps_available=torch.backends.mps.is_available(),
        packages={name: metadata.version(name) for name in ("torch", "transformers", "numpy")},
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        git_dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout),
        utc=datetime.now(timezone.utc).isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(description="Mac/CPU post-training: SFT → RM → DPO-norm → PPO-RLVR")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check device and perform a real forward/backward operation")
    prep = commands.add_parser("prepare", help="Write the deterministic offline fixture")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run all stages with isolated checkpoints and recorded evaluations")
    run.add_argument("--config", type=Path, default=Path("configs/portable/tiny.json"))
    run.add_argument("--data", type=Path, help="JSON with sft/preference/rlvr/validation/test arrays")
    run.add_argument("--output", type=Path, required=True, help="Must not already exist")
    run.add_argument("--device", choices=("auto", "cpu", "mps"))
    run.add_argument("--seed", type=int)
    infer = commands.add_parser("generate", help="Generate from a saved stage, offline")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    infer.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.command == "doctor":
        info = environment()
        for name in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
            tensor = torch.ones(4, device=name, requires_grad=True)
            tensor.square().sum().backward()
            assert tensor.grad is not None
            info[name + "_backward_ok"] = bool(torch.equal(tensor.grad.cpu(), torch.full((4,), 2.0)))
        print(json.dumps(info, indent=2))
    elif args.command == "prepare":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(data.load(), stream, indent=2)
    elif args.command == "generate":
        device = model.device_for(args.device)
        lm, tokenizer = model.restore(args.checkpoint, device)
        length = len(data.prompt_tokens(tokenizer, args.prompt))
        if args.max_new_tokens <= 0 or length + args.max_new_tokens > lm.config.max_position_embeddings:
            parser.error("Generation budget exceeds context or is nonpositive")
        print(model.generate(lm, tokenizer, args.prompt, args.max_new_tokens, device)[2])
    else:
        cfg = train.Config(**json.loads(args.config.read_text()))
        for field in ("device", "seed"):
            if getattr(args, field) is not None:
                setattr(cfg, field, getattr(args, field))
        info = environment()
        info["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("open_instruct/portable").glob("*.py"))
        }
        result = train.run(cfg, data.load(args.data), args.output)
        result["environment"] = info
        result["config"] = asdict(cfg)
        result["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        result["scope"] = "Synthetic workflow validation; not a paper benchmark reproduction"
        result["status"] = "completed"
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
