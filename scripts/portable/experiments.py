"""Run a frozen, sequential Mac validation matrix. Existing completed runs are verified and reused."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/apple_silicon")
    parser.add_argument("--include-pretrained", action="store_true", help="Downloads the pinned 135M model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "configs/portable/tiny.json").read_text())
    cases = [(f"full-mps-seed{seed}", dict(device="mps", seed=seed)) for seed in (1, 2, 3)]
    cases += [
        ("full-cpu-seed1", dict(device="cpu", seed=1)),
        ("ablation-no-dpo", dict(device="mps", seed=1, skip_dpo=True)),
        ("ablation-no-kl", dict(device="mps", seed=1, kl_beta=0.0)),
        ("ablation-dpo-sum", dict(device="mps", seed=1, normalize_dpo=False, dpo_beta=0.1)),
        ("ablation-random-value", dict(device="mps", seed=1, value_from_rm=False)),
    ]
    if args.include_pretrained:
        cases.append(("smollm2-mps", json.loads((ROOT / "configs/portable/smollm2.json").read_text())))
    configs = args.output / "configs"
    configs.mkdir(exist_ok=True)
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "open_instruct/portable").glob("*.py"))
    }
    for name, changes in cases:
        cfg = base | changes
        output = args.output / name
        summary = output / "summary.json"
        if summary.exists():
            saved = json.loads(summary.read_text())
            if (
                any(saved["config"].get(k) != v for k, v in cfg.items())
                or saved["environment"].get("source_sha256") != sources
            ):
                raise RuntimeError(f"Existing run has different config/source: {output}; choose a fresh output root")
            print(f"Verified completed run: {name}", flush=True)
            continue
        config = configs / f"{name}.json"
        config.write_text(json.dumps(cfg, indent=2))
        command = [
            sys.executable,
            "-m",
            "open_instruct.portable",
            "run",
            "--config",
            str(config),
            "--output",
            str(output),
        ]
        print(f"Running {name}", flush=True)
        started = time.monotonic()
        with (
            (args.output / f"{name}-console.txt").open("w") as console,
            (args.output / f"{name}-terminal.jsonl").open("w") as recording,
        ):
            recording.write(json.dumps(dict(seconds=0, text="$ " + " ".join(command))) + "\n")
            with subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as child:
                for line in child.stdout:
                    console.write(line)
                    console.flush()
                    recording.write(json.dumps(dict(seconds=time.monotonic() - started, text=line.rstrip())) + "\n")
                    recording.flush()
                if child.wait() != 0:
                    raise RuntimeError(f"Run failed: {name}; see its console log")
        print(f"Completed {name}: {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
