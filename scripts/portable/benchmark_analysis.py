"""Paper-relevant macro MATH and loose/instruction IFEval summaries from saved predictions."""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.portable.real_experiments import sha

from open_instruct import math_utils
from open_instruct.portable import data

BASE = Path(__file__).resolve().parents[2] / "experiments/tulu3_real"


def read(path):
    return json.loads(path.read_text())


def main():
    folder = BASE / "benchmark-rlvr-128"
    for task in ("gsm8k", "math", "ifeval"):
        if not (folder / f"{task}-summary.json").exists():
            raise RuntimeError("Finish all benchmark banks before subgroup analysis")
    metadata = read(BASE / "math-subjects.json")
    assert metadata["source_sha256"] == read(BASE / "data-v2/manifest.json")["sources"]["math"]["files"][0]["sha256"]
    predictions = [json.loads(line) for line in (folder / "math-predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 500
    groups = {"subject": defaultdict(list), "level": defaultdict(list)}
    copied_expression_passes = []
    for row in predictions:
        entry = metadata["rows"][data.digest(row["prompt"])]
        for kind in groups:
            groups[kind][str(entry[kind])].append(row["correct"])
        text = row["completion"]
        latex = re.findall(r"\$([^$]+)\$", text)
        if (
            row["correct"]
            and latex
            and not math_utils.last_boxed_only_string(text)
            and math_utils.normalize_final_answer(math_utils.get_unnormalized_answer(text)) == "[invalidanswer]"
            and latex[-1] in re.findall(r"\$([^$]+)\$", row["prompt"])
        ):
            copied_expression_passes.append(
                dict(prompt=row["prompt"], completion=text, extracted_latex=latex[-1], row_sha256=row["row_sha256"])
            )
    grouped = {
        kind: {
            key: dict(count=len(values), correct=sum(values), accuracy=statistics.mean(values))
            for key, values in sorted(buckets.items())
        }
        for kind, buckets in groups.items()
    }
    math = dict(
        micro_accuracy=statistics.mean(p["correct"] for p in predictions),
        subject_macro_accuracy=statistics.mean(v["accuracy"] for v in grouped["subject"].values()),
        **grouped,
        copied_expression_passes=copied_expression_passes,
    )
    bank = read(BASE / "data-v2/benchmarks.json")["ifeval"]
    predictions = [json.loads(line) for line in (folder / "ifeval-predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len(bank) == 541
    instructions = defaultdict(list)
    for row, prediction in zip(bank, predictions, strict=True):
        assert prediction["row_sha256"] == data.digest(row)
        for name, strict, loose in zip(
            row["constraints"]["instruction_id"],
            prediction["instruction_strict"],
            prediction["instruction_loose"],
            strict=True,
        ):
            instructions[name].append((strict, loose))
    by_instruction = {
        name: dict(
            count=len(values),
            strict=statistics.mean(v[0] for v in values),
            loose=statistics.mean(v[1] for v in values),
        )
        for name, values in sorted(instructions.items())
    }
    all_checks = [pair for values in instructions.values() for pair in values]
    ifeval = dict(
        prompt_strict=statistics.mean(p["correct"] for p in predictions),
        prompt_loose=statistics.mean(p["prompt_loose"] for p in predictions),
        instruction_strict=statistics.mean(v[0] for v in all_checks),
        instruction_loose=statistics.mean(v[1] for v in all_checks),
        by_instruction=by_instruction,
    )
    result = dict(
        math=math,
        ifeval=ifeval,
        inputs_sha256={
            str(p.relative_to(BASE)): sha(p)
            for p in (
                BASE / "math-subjects.json",
                BASE / "data-v2/benchmarks.json",
                BASE / "main-seed1/evaluations.json",
                folder / "gsm8k-predictions.jsonl",
                folder / "math-predictions.jsonl",
                folder / "ifeval-predictions.jsonl",
            )
        },
        script_sha256=sha(Path(__file__)),
    )
    panel = read(BASE / "main-seed1/evaluations.json")["rlvr"]["test"]["predictions"]
    panel_cases = []
    for path in sorted(BASE.glob("main-seed*/evaluations.json")):
        for row in read(path)["rlvr"]["test"]["predictions"]:
            if row["task"] == "math" and row["correct"]:
                candidates = re.findall(r"\$([^$]+)\$", row["completion"])
                panel_cases.append(
                    dict(
                        run=path.parent.name,
                        copied_prompt_expression=bool(
                            candidates and candidates[-1] in re.findall(r"\$([^$]+)\$", row["prompt"])
                        ),
                        prompt=row["prompt"],
                        completion=row["completion"],
                        gold_answer=row["answer"],
                    )
                )
    (BASE / "verifier-audit-panel.json").write_text(
        json.dumps(
            dict(note="Automated last-LaTeX/prompt comparison; frozen scores retained.", cases=panel_cases), indent=2
        )
    )
    full = {}
    for task in ("gsm8k", "math", "ifeval"):
        for line in (folder / f"{task}-predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            full[data.digest(row["prompt"])] = row
    mismatch = [
        dict(
            prompt=row["prompt"],
            panel_correct=row["correct"],
            benchmark_correct=full[data.digest(row["prompt"])]["correct"],
            same_text=row["completion"] == full[data.digest(row["prompt"])]["completion"],
        )
        for row in panel
        if row["completion"] != full[data.digest(row["prompt"])]["completion"]
        or row["correct"] != full[data.digest(row["prompt"])]["correct"]
    ]
    result["panel_consistency"] = dict(
        count=len(panel), mismatches=mismatch, panel_batch_size=4, benchmark_batch_size=8
    )
    (BASE / "benchmark-analysis.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Benchmark subgroup analysis",
        "",
        "These metrics use the saved 128-token, zero-shot predictions. They remain a reduced protocol.",
        "",
        f"MATH-500 micro accuracy: {math['micro_accuracy']:.2%}; equal-weight subject macro accuracy: **{math['subject_macro_accuracy']:.2%}**.",
        "",
        "| MATH subject | Correct / count | Accuracy |",
        "|---|---:|---:|",
    ]
    for name, values in math["subject"].items():
        lines.append(f"| {name} | {values['correct']}/{values['count']} | {values['accuracy']:.2%} |")
    lines += ["", "| MATH difficulty | Correct / count | Accuracy |", "|---|---:|---:|"]
    for name, values in math["level"].items():
        lines.append(f"| {name} | {values['correct']}/{values['count']} | {values['accuracy']:.2%} |")
    lines += [
        "",
        f"IFEval prompt strict: {ifeval['prompt_strict']:.2%}; prompt loose: **{ifeval['prompt_loose']:.2%}**. Instruction strict: {ifeval['instruction_strict']:.2%}; loose: {ifeval['instruction_loose']:.2%}.",
        "",
        "| Instruction type | Count | Strict | Loose |",
        "|---|---:|---:|---:|",
    ]
    for name, values in by_instruction.items():
        lines.append(f"| {name} | {values['count']} | {values['strict']:.2%} | {values['loose']:.2%} |")
    lines += [
        "",
        f"Stage-panel consistency: {len(panel) - len(mismatch)}/{len(panel)} overlapping questions have identical text and correctness "
        "between batch-4 stage evaluation and batch-8 full-bank evaluation. Any differences are retained in the analysis JSON.",
        "",
        f"Extraction audit: **{len(copied_expression_passes)}** MATH passes use a last-LaTeX candidate appearing verbatim in the question, "
        "with neither a boxed nor Minerva final answer. These are flagged for interpretation, not silently rescored. "
        "Symbolic equivalence can give credit to repeating an equivalent expression without producing the requested answer; "
        "see the saved cases in `benchmark-analysis.json`.",
        "",
        "[Tülu 3 §7.2](https://arxiv.org/html/2411.15124v5#S7.SS2) uses MATH subject macro averages and IFEval loose prompt accuracy. Matching the aggregation does not match the model, question subset, prompts or output budget. Instruction checks sharing a prompt are not statistically independent.",
        "",
    ]
    (BASE / "BENCHMARK_ANALYSIS.md").write_text("\n".join(lines))
    print("Saved MATH subject/level and IFEval instruction breakdowns")


if __name__ == "__main__":
    main()
