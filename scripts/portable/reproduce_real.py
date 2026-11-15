"""Reproduce the real-data study in a fresh workspace without relying on ignored recorded weights."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "output"):
        raise ValueError("Use a new directory under output/ or outside the repository")
    paths = []
    for folder, pattern in (
        ("open_instruct/portable", "*.py"),
        ("open_instruct/IFEvalG", "*.py"),
        ("scripts/portable", "*.py"),
        ("configs/portable", "*.json"),
        ("tests/portable", "*.py"),
        ("experiments/tulu3_real/data-v2", "*"),
    ):
        paths += [p for p in (ROOT / folder).glob(pattern) if p.is_file()]
    paths += [
        ROOT / name
        for name in (
            "open_instruct/__init__.py",
            "open_instruct/logger_utils.py",
            "open_instruct/math_utils.py",
            "open_instruct/if_functions.py",
            "requirements-mac.txt",
            "scripts/portable/setup.sh",
            "Makefile",
            "README.md",
            "README.zh-CN.md",
            "LICENSE",
            "docs/apple_silicon.md",
            "docs/tulu3_mac_experiments.md",
            "experiments/tulu3_real/math-subjects.json",
            "experiments/tulu3_real/branch-rng-protocol.json",
        )
    ]
    paths += [
        p
        for p in (ROOT / "experiments/tulu3_real/source-v4").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    destination.mkdir(parents=True, exist_ok=False)
    receipts = {}
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        receipts[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "workspace-manifest.json").write_text(
        json.dumps(
            dict(
                source_workspace=str(ROOT),
                files_sha256=receipts,
                note="Fresh copied source/data workspace; no recorded run summaries or checkpoints are reused.",
            ),
            indent=2,
        )
    )
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "output/tulu3-reproduction")
    parser.add_argument("--prepare-only", action="store_true", help="Copy and verify inputs without starting training")
    args = parser.parse_args()
    workspace = prepare(args.workspace)
    print(f"Prepared fresh workspace: {workspace}", flush=True)
    if args.prepare_only:
        return
    for module, arguments in (
        ("real_experiments", ["suite"]),
        ("preference_holdout", []),
        ("decode_sensitivity", []),
        ("benchmark_analysis", []),
        ("report_real", []),
        ("audit_real", []),
        ("real_demo", []),
    ):
        print(f"Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.portable." + module, *arguments], cwd=workspace, check=True)
    print(f"Completed. Report: {workspace / 'experiments/tulu3_real/REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
