"""Render an explicitly labeled replay from completed real-data experiments, including failures in predictions."""

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"


def read(path):
    return json.loads(path.read_text())


def main():
    aggregate = read(BASE / "aggregate.json")
    if aggregate["pending"] or len(aggregate["benchmarks"]) != 3:
        raise RuntimeError("Finish the planned training and three full benchmark banks before rendering")
    target = ROOT / "assets/demo"
    target.mkdir(exist_ok=True, parents=True)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default(size=19)
    heading = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default(size=28)
    pages = []

    def page(title, lines):
        canvas = Image.new("RGB", (1120, 640), "#0c1526")
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 25), "TULU 3 WORKFLOW / M1 PRO", font=heading, fill="#69e5b9")
        draw.text((32, 78), title, font=font, fill="#f8d788")
        y = 125
        for line in lines:
            for wrapped in textwrap.wrap(line, 90) or [""]:
                if y > 545:
                    break
                draw.text((32, y), wrapped, font=font, fill="#d9e5fa")
                y += 28
        draw.text(
            (32, 598), "Recorded results replay / 135M model / not paper-scale reproduction", font=font, fill="#91a2bd"
        )
        pages.append(canvas)

    page(
        "Real published data, complete local stages",
        [
            "SFT -> reward model -> DPO-norm -> PPO-RLVR -> task evaluation",
            "96 SFT examples / 48 preference pairs / 72 available RL prompts",
            "24 RL prompts consumed per training run; fixed short budgets",
            f"Completed: {len(aggregate['seeds'])} full seeds and {len(aggregate['branches'])} anchored branches",
            "Original dataset revisions, row hashes, losses, rollouts and checkpoints are saved.",
            "This video replays saved measurements. It is not a live screen recording.",
        ],
    )
    records = [json.loads(line) for line in (BASE / "main-seed1/metrics.jsonl").read_text().splitlines()]
    lines = []
    for stage in ("sft", "reward_model", "dpo", "rlvr"):
        row = [r for r in records if r["stage"] == stage][-1]
        lines.append(f"{stage:14s} final step {row['step']:3d}   recorded loss {row['loss']:.6f}")
    lines += [
        "",
        "Loss values have different objectives; they are not accuracy scores.",
        "See REPORT.md for negative results, EOS penalties and seed variation.",
    ]
    page("Actual final optimization records / seed 1", lines)
    lines = ["All questions evaluated with zero-shot prompts and a 128-token cap.", ""]
    for path, summary in aggregate["benchmarks"].items():
        task = Path(path).name.replace("-summary.json", "")
        lines.append(
            f"{task:8s} {summary['correct']:4d}/{summary['count']:<4d} correct   accuracy {summary['accuracy']:.2%}   EOS {summary['eos_rate']:.2%}"
        )
    lines += [
        "",
        "IFEval uses strict prompt-level accuracy. MATH-500 is a subset of MATH.",
        "Full question coverage does not imply full paper-protocol parity.",
    ]
    page("Measured final benchmark results", lines)
    for task in ("gsm8k", "ifeval"):
        row = json.loads((BASE / f"benchmark-rlvr-128/{task}-predictions.jsonl").read_text().splitlines()[0])
        page(
            f"Actual saved prediction / first {task} row",
            [
                "PROMPT: " + str(row["prompt"])[:280],
                "",
                "RESPONSE (excerpt): " + row["completion"][:420],
                "",
                f"Verifier passed: {row['correct']} / EOS: {row['eos']} / generated tokens: {row['tokens']}",
            ],
        )
    pages[0].save(target / "tulu3-real.gif", save_all=True, append_images=pages[1:], duration=6000, loop=0)
    with tempfile.TemporaryDirectory() as directory:
        for i, frame in enumerate(pages):
            frame.save(Path(directory) / f"frame-{i:02d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "1/6",
                "-i",
                str(Path(directory) / "frame-%02d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "24",
                "-movflags",
                "+faststart",
                str(target / "tulu3-real.mp4"),
            ],
            check=True,
        )
    (target / "tulu3-real.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tulu 3 real-data experiment replay</title>
<style>body{max-width:1120px;margin:3rem auto;padding:1rem;background:#0c1526;color:#d9e5fa;font:18px system-ui}video{width:100%}a{color:#69e5b9}</style>
<h1>Tülu 3 workflow on M1 Pro</h1><p>Saved real-data results and fixed first-row predictions. Reduced-scale protocol; not a live screen recording.</p>
<video autoplay muted loop playsinline controls><source src="tulu3-real.mp4" type="video/mp4"></video>
<p>Browser settings may block autoplay; playback controls remain available. <a href="tulu3-real.gif">Looping GIF</a></p></html>""")
    print("Rendered", len(pages) * 6, "seconds from completed experiment artifacts")


if __name__ == "__main__":
    main()
