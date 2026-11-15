"""Rebuild the report, figures and demo exclusively from saved experiment evidence."""

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import tempfile
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def make_report(root, runs):
    lines = [
        "# Apple Silicon experiment report",
        "",
        "These are actual local executions, not Tülu benchmark reproductions. The default experiment uses "
        "synthetic addition/copy tasks and a randomly initialized tiny GPT-2 architecture. "
        "The pretrained run uses the same fixture to test the Hugging Face model path.",
        "",
        "## Measurement protocol",
        "",
        "The split seed is fixed at 42: 200 SFT, 80 preference/RM, 56 RLVR, 32 validation and 32 test prompts. "
        "All five sets have disjoint normalized prompts. No hyperparameters or checkpoints are selected using test scores; "
        "the final step of every stage is evaluated. This is exact-prompt decontamination, not semantic or n-gram decontamination. "
        "Each exact-match test point is 1/32 (3.125 percentage points). The tasks do not measure general instruction following, safety or reasoning.",
        "",
        "Experiments run sequentially to avoid competing device workloads. Runtime below includes all stage evaluations "
        "and checkpoint writes, but excludes model construction/download. Peak RSS is the process high-water mark, "
        "not total system or Metal memory. The host was not exclusively reserved, so timings are descriptive. CPU/MPS trajectories need not be bitwise identical.",
        "",
        "| Run | Device | Parameters | Seconds | Peak process RSS GiB | SFT EM | DPO EM | RLVR EM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        scores = run["evaluations"]
        dpo = scores.get("dpo", scores.get("dpo_skipped"))["test"]["exact_match"]
        lines.append(
            f"| [{name}]({name}/summary.json) | {run['device']} | {run['parameters']:,} | "
            f"{run['elapsed_seconds']:.1f} | {run['peak_process_rss_bytes'] / 2**30:.2f} | "
            f"{scores['sft']['test']['exact_match']:.2%} | {dpo:.2%} | {scores['rlvr']['test']['exact_match']:.2%} |"
        )
    full = [runs[f"full-mps-seed{s}"] for s in (1, 2, 3) if f"full-mps-seed{s}" in runs]
    lines += [
        "",
        "## Across-seed results",
        "",
        "Arithmetic mean ± sample standard deviation across the three MPS seeds; "
        "these are descriptive statistics, not confidence intervals.",
        "",
        "| Stage | Test exact match | Preference accuracy | Mean response NLL |",
        "|---|---:|---:|---:|",
    ]
    for stage in ("base", "sft", "dpo", "rlvr"):
        values = [
            [run["evaluations"][stage]["test"][metric] for run in full]
            for metric in ("exact_match", "preference_accuracy", "response_nll")
        ]
        if len(full) >= 2:
            formatted = [f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}" for v in values]
            lines.append(f"| {stage} | " + " | ".join(formatted) + " |")
    if full:
        deltas = [
            run["evaluations"]["rlvr"]["test"]["exact_match"] - run["evaluations"]["dpo"]["test"]["exact_match"]
            for run in full
        ]
        lines += [
            "",
            f"The mean RLVR-minus-DPO exact-match change is {100 * statistics.mean(deltas):+.2f} percentage points. "
            "A change on this small synthetic sample does not establish a general RLVR improvement. "
            "Preference likelihood and free-generation correctness measure different things; lower loss alone is not sufficient.",
        ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- The stage checkpoints, finite optimization traces and successful reload tests demonstrate executable training, "
        "rather than a simulated loss curve. The base model in the tiny runs has no pretraining knowledge.",
        "- SFT receives correct targets, DPO compares chosen/rejected likelihoods against an immutable SFT cache, "
        "and PPO receives only sampled-response verifier rewards plus a reference-policy KL penalty. "
        "Missing EOS incurs a separate penalty, so a negative mean reward can coexist with some verified answers.",
        "- Ablations remove DPO, remove KL, use summed DPO with beta=0.1, or initialize the critic from the policy "
        "with a fresh head instead of the trained RM. Each uses seed 1 only; the DPO-sum comparison changes beta too. "
        "They are diagnostic comparisons, not controlled estimates of independent causal effects. "
        "Even equal-seed MPS runs can have different SFT baselines due to numerical nondeterminism. Removing DPO also changes later RNG consumption, so generated trajectories are not paired.",
        "- CPU/MPS timing uses one CPU run and one comparable MPS run on this machine. "
        "Very small models can favor CPU because accelerator dispatch costs dominate. Do not extrapolate to large models.",
        "- No all-task OLMES suite, judge-model evaluation, published dataset mixtures, 8B/70B/405B scaling "
        "or paper-wide data ablations were run. [Coverage audit](../../docs/apple_silicon.md#paper-coverage) lists these gaps.",
        "",
        "## Evidence and rerunning",
        "",
        "Each run contains `config.json`, `dataset.json`, `metrics.jsonl`, `rollouts.jsonl`, "
        "`dpo_reference.json` (unless skipped), `evaluations.json` and `summary.json`. Summaries include "
        "package versions, source hashes, device, timing and revision. Checkpoints are present locally but git-ignored. "
        "`environment-lock.txt` records this Mac's installed packages; it is not a cross-platform lockfile.",
        "",
        "```bash",
        ".venv-mac/bin/python scripts/portable/experiments.py --include-pretrained --output output/mac-reproduction",
        ".venv-mac/bin/python scripts/portable/report.py --input output/mac-reproduction --demo-dir output/mac-demo-media",
        "```",
        "",
        "![Measured stage metrics and training traces](metrics.png)",
        "",
        "The earlier `mps-seed1` directory is a 120-step SFT pilot retained for transparency; "
        "it is excluded from the comparison table and aggregate statistics. The main preset uses 600 SFT steps.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    (root / "aggregate.json").write_text(json.dumps(runs, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    stages = ["base", "sft", "dpo", "rlvr"]
    for seed, run in enumerate(full, 1):
        for ax, metric, label in [
            (axes[0, 0], "exact_match", "Test exact match"),
            (axes[0, 1], "preference_accuracy", "Test preference accuracy"),
        ]:
            ax.plot(
                stages,
                [run["evaluations"][stage]["test"][metric] for stage in stages],
                marker="o",
                label=f"seed {seed}",
            )
            ax.set_title(label)
            ax.set_ylim(0, 1)
            ax.legend()
        records = [
            json.loads(line) for line in (root / f"full-mps-seed{seed}" / "metrics.jsonl").read_text().splitlines()
        ]
        sft = [r for r in records if r["stage"] == "sft"]
        rlvr = [r for r in records if r["stage"] == "rlvr" and r["epoch"] == 0]
        axes[1, 0].plot([r["step"] for r in sft], [r["loss"] for r in sft], alpha=0.7)
        axes[1, 1].plot([r["step"] for r in rlvr], [r["verified_rate"] for r in rlvr], alpha=0.7)
    axes[1, 0].set_title("Actual SFT optimization loss")
    axes[1, 1].set_title("Actual rollout verified fraction (batch of 4)")
    for ax in axes[1]:
        ax.set_xlabel("Optimization step")
    fig.suptitle("M1 Pro / synthetic fixture — not paper benchmark scores")
    fig.tight_layout()
    fig.savefig(root / "metrics.png", dpi=160)
    plt.close(fig)


def make_demo(root, target):
    target.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (root / "full-mps-seed1-terminal.jsonl").read_text().splitlines()]
    summary = read(root / "full-mps-seed1/summary.json")
    evaluations = read(root / "full-mps-seed1/evaluations.json")
    font_path = "/System/Library/Fonts/Menlo.ttc"
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default(size=18)
    heading = ImageFont.truetype(font_path, 27) if Path(font_path).exists() else ImageFont.load_default(size=27)
    frames = []

    def frame(title, lines, footer):
        canvas = Image.new("RGB", (1120, 640), "#0c1526")
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 26), "OPEN INSTRUCT  /  Apple Silicon", font=heading, fill="#69e5b9")
        draw.text((32, 72), title, font=font, fill="#f8d788")
        y = 120
        for line in lines:
            for wrapped in textwrap.wrap(line, 94, replace_whitespace=False) or [""]:
                draw.text((32, y), wrapped, font=font, fill="#d9e5fa")
                y += 25
                if y > 550:
                    break
            if y > 550:
                break
        draw.text((32, 600), footer, font=font, fill="#91a2bd")
        frames.append(canvas)

    frame(
        "Recorded execution / M1 Pro 16 GB / PyTorch MPS",
        [
            "$ python -m open_instruct.portable run --device mps ...",
            f"Model: {summary['parameters']:,} parameters, randomly initialized causal Transformer",
            "Data: synthetic addition and copy; disjoint training / validation / test",
            "Sequence: SFT -> reward model -> normalized DPO -> PPO-RLVR -> evaluation",
            "This is a time-compressed replay of actual saved logs and predictions.",
            "Not a paper-scale reproduction or a general-purpose chatbot.",
        ],
        "Source: experiments/apple_silicon/full-mps-seed1/",
    )
    selected = []
    for r in records:
        text = r["text"]
        if " - INFO - " not in text:
            continue
        if (
            "test exact match" in text
            or "step 1:" in text
            or "step 200:" in text
            or "step 400:" in text
            or "step 600:" in text
            or "step 60:" in text
            or ("rlvr step 40:" in text and "'epoch': 3" in text)
        ):
            if "rlvr" in text and "'epoch':" in text and not ("'epoch': 0" in text or "'epoch': 3" in text):
                continue
            selected.append(r)
    recent = []
    for r in selected:
        recent.append(r["text"].split(" - INFO - ")[-1])
        frame(
            f"Recorded elapsed time: {r['seconds']:.1f} seconds",
            recent[-5:],
            "Actual training output / playback compressed / no invented metrics",
        )
    for stage in ("base", "sft", "dpo", "rlvr"):
        result = evaluations[stage]["test"]
        lines = [f"Held-out exact match: {result['exact_match']:.2%} over {result['count']} prompts", ""]
        for prediction in result["predictions"][:4]:
            lines += [
                prediction["prompt"],
                f"  Model: {prediction['completion']!r} | Expected: {prediction['answer']}",
            ]
        frame(
            f"Checkpoint: {stage} / first four test rows (fixed order)",
            lines,
            "Predictions from evaluations.json / includes model errors",
        )
    frames[0].save(target / "portable-demo.gif", save_all=True, append_images=frames[1:], duration=1000, loop=0)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg to produce the MP4; GIF was generated")
    with tempfile.TemporaryDirectory(prefix="open-instruct-demo-") as directory:
        for index, canvas in enumerate(frames):
            canvas.save(Path(directory) / f"{index:03d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                "1",
                "-i",
                str(Path(directory) / "%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "24",
                "-movflags",
                "+faststart",
                str(target / "portable-demo.mp4"),
            ],
            check=True,
        )
    (target / "index.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Instruct — M1 demo</title>
<style>body{margin:3rem auto;max-width:1120px;padding:0 1rem;background:#0c1526;color:#d9e5fa;font:18px system-ui}video{width:100%}a{color:#69e5b9}</style>
<h1>Open Instruct on Apple Silicon</h1><p>Time-compressed replay of an actual M1 Pro run. Synthetic tasks; not paper benchmarks.</p>
<video autoplay muted loop playsinline controls aria-label="Recorded M1 Pro training and predictions"><source src="portable-demo.mp4" type="video/mp4"></video>
<p>Browser settings may prevent autoplay; use the play control. <a href="portable-demo.gif">Animated GIF</a> · <a href="../../README.md">README</a></p></html>""")
    (target / "provenance.json").write_text(
        json.dumps(
            dict(
                recording=str((root / "full-mps-seed1-terminal.jsonl").relative_to(ROOT))
                if root.is_relative_to(ROOT)
                else str(root),
                recording_sha256=hashlib.sha256((root / "full-mps-seed1-terminal.jsonl").read_bytes()).hexdigest(),
                frames=len(frames),
                frame_duration_seconds=1,
                kind="Time-compressed replay of recorded execution and predictions",
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("portable-demo.*")},
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "experiments/apple_silicon")
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "assets/demo")
    args = parser.parse_args()
    runs = {
        p.parent.name: read(p)
        for p in sorted(args.input.glob("*/summary.json"))
        if p.parent.name != "mps-seed1" and read(p).get("status") == "completed"
    }
    if not runs:
        raise RuntimeError("No completed runs found")
    make_report(args.input, runs)
    make_demo(args.input, args.demo_dir)


if __name__ == "__main__":
    main()
