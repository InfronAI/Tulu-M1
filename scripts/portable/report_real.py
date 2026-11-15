"""Summarize only completed real-data experiments; incomplete runs remain explicitly incomplete."""

import json
import math
import statistics
from pathlib import Path

from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/tulu3_real"


def read(path):
    return json.loads(path.read_text())


def wilson(correct, count):
    z = 1.95996398454
    p = correct / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return max(0.0, center - half), min(1.0, center + half)


def paired(control, variant):
    if [r["prompt"] for r in control] != [r["prompt"] for r in variant]:
        raise ValueError("Paired comparisons require exactly aligned test prompts")
    wins = sum(not a["correct"] and b["correct"] for a, b in zip(control, variant))
    losses = sum(a["correct"] and not b["correct"] for a, b in zip(control, variant))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return dict(wins=wins, losses=losses, exact_mcnemar_p=p)


def main():
    dataset_manifest = read(BASE / "data-v2/manifest.json")
    plan = read(BASE / "plan.json")
    seeds = {
        p.parent.name: read(p)
        for p in sorted(BASE.glob("main-seed*/summary.json"))
        if read(p).get("status") == "completed"
    }
    branches = {
        name: read(BASE / name / "summary.json")
        for name in plan["branches"]
        if (BASE / name / "summary.json").exists()
    }
    benchmarks = {str(p.relative_to(BASE)): read(p) for p in sorted(BASE.glob("benchmark-*/*-summary.json"))}
    sensitivity = {p.name: read(p) for p in sorted((BASE / "decode-sensitivity").glob("*-summary.json"))}
    analysis_path = BASE / "benchmark-analysis.json"
    analysis = read(analysis_path) if analysis_path.exists() else None
    panel_audit_path = BASE / "verifier-audit-panel.json"
    panel_cases = read(panel_audit_path)["cases"] if panel_audit_path.exists() else []
    holdout_path = BASE / "preference-holdout-results.json"
    holdout = read(holdout_path) if holdout_path.exists() else {"runs": {}}
    expected = [f"main-seed{s}" for s in plan["seeds"]] + list(plan["branches"])
    pending = [name for name in expected if name not in seeds and name not in branches]
    lines = [
        "# Tülu 3 real-data experiments on M1 Pro",
        "",
        "**Scope:** actual published training examples and real task questions, SmolLM2-135M, short local budgets. "
        "This is not a reproduction of Tülu 3 published scores or its original model scale.",
        "",
        f"Completed: {len(seeds)}/3 full training seeds, {len(branches)}/{len(plan['branches'])} anchored RL branches. "
        f"Pending/incomplete training runs: {', '.join(pending) if pending else 'none'}.",
        "",
        "[Detailed protocol and deviations](../../docs/tulu3_mac_experiments.md) · [Frozen plan](plan.json) · "
        "[Data provenance](data-v2/manifest.json) · [Runtime](environment-lock.txt)",
        "",
        "## Data coverage",
        "",
        f"Training: {dataset_manifest['split_sizes']['sft']} SFT rows across {len(dataset_manifest['sft_sources'])} sources, "
        f"{dataset_manifest['split_sizes']['preference']} unchanged preference pairs across {len(dataset_manifest['preference_sources'])} sources, "
        "72 RL prompts (24 per domain), 16 unused preference pairs, 24 validation and 72 test questions. "
        "Data come from pinned revisions of every SFT/preference shard. The scan and length-filtered source-balanced selection "
        "are bounded and differ from original mixture proportions. Exact normalized prompts and 13-word spans "
        "are screened against all 2,360 evaluation questions.",
        "",
        "## Stage-by-stage results (72-question fixed panel)",
        "",
        "| Run | Stage | GSM8K / 24 | MATH-500 / 24 | IFEval strict / 24 | IFEval loose | EOS rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in seeds.items():
        for stage, sets in summary["evaluations"].items():
            tasks = sets["test"]["per_task"]
            eos = sum(t["eos_rate"] * t["count"] for t in tasks.values()) / sum(t["count"] for t in tasks.values())
            lines.append(
                f"| [{name}]({name}/summary.json) | {stage} | {tasks['gsm8k']['accuracy']:.2%} | "
                f"{tasks['math']['accuracy']:.2%} | {tasks['ifeval']['accuracy']:.2%} | "
                f"{tasks['ifeval']['prompt_loose']:.2%} | {eos:.2%} |"
            )
    lines += [
        "",
        "Test-panel resolution is 4.17 percentage points per domain. Final-step checkpoints are fixed in advance; "
        "no best-of-test checkpoint selection was performed. Raw predictions, including failures and incomplete answers, are saved.",
        "",
        "## Across-seed variation",
        "",
        "| Stage | GSM8K mean ± SD | MATH mean ± SD | IF strict mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    if len(seeds) >= 2:
        for stage in ("base", "sft", "dpo", "rlvr"):
            cells = []
            for task in ("gsm8k", "math", "ifeval"):
                values = [s["evaluations"][stage]["test"]["per_task"][task]["accuracy"] for s in seeds.values()]
                cells.append(f"{statistics.mean(values):.2%} ± {statistics.stdev(values):.2%}")
            lines.append("| " + stage + " | " + " | ".join(cells) + " |")
    else:
        lines += ["", "Fewer than two completed seeds; no across-seed variance estimate yet."]
    if seeds:
        base_accuracy = statistics.mean(s["evaluations"]["base"]["test"]["exact_match"] for s in seeds.values())
        final_accuracy = statistics.mean(s["evaluations"]["rlvr"]["test"]["exact_match"] for s in seeds.values())
        lines += [
            "",
            f"Observed across completed seeds: mean overall panel accuracy changed from {base_accuracy:.2%} "
            f"at the base checkpoint to {final_accuracy:.2%} after RLVR ({100 * (final_accuracy - base_accuracy):+.2f} percentage points). "
            "This measurement describes the fixed small panel and reduced training recipe; it is not a claim about the original Tülu 3 method.",
        ]
    lines += [
        "",
        "## Checkpoint-anchored ablations",
        "",
        "Compare against the **control fork**, not the main seed-1 trajectory: every fork restarts the same RL RNG and "
        "fresh optimizer. Anchor hashes are in each summary. The no-DPO and DPO-sum recipes intentionally use the SFT anchor; "
        "other variants must have exactly the control DPO anchor. Changing policy or prompt distribution changes future samples.",
        "",
        "| Branch | GSM8K | MATH | IF strict | Paired wins/losses vs control | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparisons = {}
    control = read(BASE / "control/evaluations.json")["predictions"] if "control" in branches else None
    for name, summary in branches.items():
        metrics = summary["evaluation"]["per_task"]
        comparison = None
        if control is not None:
            if name not in ("no-dpo", "dpo-sum"):
                assert summary["anchor_sha256"] == branches["control"]["anchor_sha256"]
            comparison = paired(control, read(BASE / name / "evaluations.json")["predictions"])
            comparisons[name] = comparison
        cells = [f"{metrics[t]['accuracy']:.2%}" for t in ("gsm8k", "math", "ifeval")]
        cells += [
            f"{comparison['wins']}/{comparison['losses']}" if comparison else "pending",
            f"{comparison['exact_mcnemar_p']:.4f}" if comparison else "pending",
        ]
        lines.append(f"| [{name}]({name}/summary.json) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "These p-values are exploratory and uncorrected for multiple comparisons. Single-seed forks and tiny panels "
        "cannot establish a robust algorithm ranking. Summed DPO changes both beta and normalization.",
        "",
        "## Full question coverage under a bounded decoding budget",
        "",
        "| Result | Correct / total | Accuracy (Wilson 95% interval) | EOS rate | Symbolic timeouts / length skips |",
        "|---|---:|---:|---:|---:|",
    ]
    for path, summary in benchmarks.items():
        low, high = wilson(summary["correct"], summary["count"])
        lines.append(
            f"| [{path}]({path}) | {summary['correct']}/{summary['count']} | {summary['accuracy']:.2%} "
            f"({low:.2%}–{high:.2%}) | {summary['eos_rate']:.2%} | "
            f"{summary['symbolic_timeouts']} / {summary['symbolic_length_skips']} |"
        )
    for task, count in [("gsm8k", 1319), ("math", 500), ("ifeval", 541)]:
        path = BASE / f"benchmark-rlvr-128/{task}-summary.json"
        if not path.exists():
            predictions = path.with_name(task + "-predictions.jsonl")
            done = len(predictions.read_text().splitlines()) if predictions.exists() else 0
            lines.append(
                f"| {task}: incomplete | {done}/{count} predictions recorded | not scored as complete | — | — |"
            )
    if analysis:
        lines += [
            "",
            f"Additional paper-relevant aggregation: MATH-500 subject macro accuracy **{analysis['math']['subject_macro_accuracy']:.2%}**; "
            f"IFEval loose prompt accuracy **{analysis['ifeval']['prompt_loose']:.2%}**. "
            "[Subject, difficulty and instruction breakdowns](BENCHMARK_ANALYSIS.md) preserve both strict and loose results. "
            "The main benchmark table above uses question-level micro accuracy (strict for IFEval).",
        ]
    lines += [
        "",
        "These intervals reflect question-level binomial sampling uncertainty, not training-seed uncertainty. "
        "For IFEval the table uses prompt-level strict accuracy; the result JSON also includes loose and instruction-level scores. "
        "Symbolic timeout/length-skip counts are candidate-comparison events, potentially multiple per response. "
        "Only the preselected final seed-1 checkpoint is evaluated on every question; these totals do not estimate full-bank improvement over the base model. "
        "**128 generated tokens and zero-shot prompts do not reproduce the paper protocol.** Long instruction requests may be "
        "impossible under this cap, and mathematical solutions may stop before reaching an answer. MATH-500 is not the full MATH test set.",
        "",
        "## Untouched preference holdout",
        "",
        "The 16 original held-out chosen/rejected pairs are scored by length-normalized response log probability. "
        "This is pairwise label agreement, not a judge-based chat win rate. Ties count as incorrect.",
        "",
        "| Run | Base | SFT | DPO | RLVR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stages in holdout["runs"].items():
        cells = [f"{stages[s]['accuracy']:.2%}" for s in ("base", "sft", "dpo", "rlvr")]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Observed RL training signal",
        "",
        "| Run | Verified rollouts | EOS rollouts | Positive terminal rewards | Mean terminal reward |",
        "|---|---:|---:|---:|---:|",
    ]
    rollout_metrics = {}
    for name in [*seeds, *branches]:
        rows = [json.loads(line) for line in (BASE / name / "rollouts.jsonl").read_text().splitlines()]
        metrics = dict(
            count=len(rows),
            verified=sum(r["verified"] for r in rows),
            eos=sum(r["eos"] for r in rows),
            positive=sum(r["reward"] > 0 for r in rows),
            mean_reward=statistics.mean(r["reward"] for r in rows),
        )
        rollout_metrics[name] = metrics
        lines.append(
            f"| {name} | {metrics['verified']:g}/{len(rows)} | {metrics['eos']}/{len(rows)} | {metrics['positive']}/{len(rows)} | {metrics['mean_reward']:.3f} |"
        )
    lines += [
        "",
        "Correctness is measured independently of EOS. Without EOS, the terminal reward is -10 even when a verifier passes, "
        "as specified by this fixed configuration. A run dominated by this penalty does not test the benefits of sustained positive verifier rewards.",
        "",
        "## Decoding-budget sensitivity",
        "",
        "A supplementary paired check generates up to 512 tokens for the same 72 test prompts at the base and final seed-1 checkpoints. "
        "It is not used for checkpoint or hyperparameter selection. More tokens can both recover answers and introduce new instruction violations.",
        "",
        "| Checkpoint | Task | Correct at 128 / 24 | Correct at 512 / 24 | Gained / lost | EOS at 512 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in sensitivity.values():
        for task, metrics in summary["per_task"].items():
            lines.append(
                f"| {summary['stage']} | {task} | {metrics['short_correct']} | {metrics['correct']} | {metrics['gained']} / {metrics['lost']} | {metrics['eos_rate']:.2%} |"
            )
    if len(sensitivity) < 2:
        lines += ["", "Supplementary decoding checks are not yet both complete; only finished results are shown."]
    lines += [
        "",
        "## Failure and limitation analysis",
        "",
        f"- The panel extraction check flags {sum(c['copied_prompt_expression'] for c in panel_cases)}/{len(panel_cases)} final-checkpoint MATH passes "
        "with a last-LaTeX expression copied from the question ([saved cases](verifier-audit-panel.json)). "
        "Repeating an equivalent expression can receive verifier credit without solving the task. "
        "The full-benchmark subgroup analysis flags analogous last-LaTeX copies without silently changing the scoring rules.",
        "- One-question gains on the small panel do not establish a robust improvement. Inspect responses for instruction echoes "
        "that can satisfy mechanical constraints without completing the requested task.",
        "- The original 2.8 MPS real-data pilot failed during SFT on nonfinite gradient norm. "
        "Its console and independent 96-step diagnostic are retained. The independent diagnostic did not reproduce that failure; "
        "the precise cause is unproven. The main matrix uses separately pinned PyTorch 2.15 and is checked for finite losses/gradients.",
        "- An eager-attention attempt reached about 12 GB physical footprint with heavy swapping and was stopped after RM. "
        "Its diagnostics and checkpoints remain separate. The completed matrix uses SDPA, final-position generation logits, "
        "cache release, activation checkpointing and an explicit MPS allocation limit; archived sources identify each attempt. "
        "The first capped SDPA attempt also stopped on RM allocation pressure; longest-example RM/DPO/PPO stress tests preceded the final matrix.",
        "- Low math scores or unchanged scores are expected possible outcomes for a 135M base model and short training budget. "
        "Successful execution is not evidence of useful reasoning capability or replication of paper gains.",
        "- A small EOS rate means missing-EOS penalties can dominate PPO. Inspect raw verifier successes separately from mean rewards. "
        "Rewards are not silently changed to make the curves look better.",
        "- Real labels replace synthetic wrong answers for DPO. The RL training IF checker follows the released legacy constraints, "
        "while the official benchmark uses the current repository instruction registry. Verifier and protocol differences remain explicit.",
        "- Published model scale, complete mixture training, all data ablations, judge/human evaluations and the full OLMES "
        "capability/safety/unseen suite remain unexecuted. No incomplete job is included in completed counts.",
        "",
        "## Reproduce and inspect",
        "",
        "```bash",
        ".venv-mac-real/bin/python -m scripts.portable.real_experiments suite",
        ".venv-mac-real/bin/python -m scripts.portable.report_real",
        "```",
        "",
        "Check `data-v2/selected_original_rows.json`, per-run `metrics.jsonl` and `rollouts.jsonl`, "
        "checkpoint hashes, saved predictions, and complete benchmark summaries. Checkpoints remain local and git-ignored.",
        "",
    ]
    (BASE / "REPORT.md").write_text("\n".join(lines))
    (BASE / "aggregate.json").write_text(
        json.dumps(
            dict(
                seeds=seeds,
                branches=branches,
                benchmarks=benchmarks,
                paired_comparisons=comparisons,
                pending=pending,
                preference_holdout=holdout,
                rollout_metrics=rollout_metrics,
                decoding_sensitivity=sensitivity,
                benchmark_analysis=analysis,
            ),
            indent=2,
        )
    )
    if seeds:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        stages = ["base", "sft", "dpo", "rlvr"]
        maximum = max(
            s["evaluations"][stage]["test"]["per_task"][task]["accuracy"]
            for s in seeds.values()
            for stage in stages
            for task in ("gsm8k", "math", "ifeval")
        )
        ceiling = min(1.0, max(0.25, math.ceil(maximum * 4) / 4))
        for ax, task in zip(axes, ("gsm8k", "math", "ifeval")):
            for name, summary in seeds.items():
                ax.plot(
                    stages,
                    [summary["evaluations"][s]["test"]["per_task"][task]["accuracy"] for s in stages],
                    marker="o",
                    label=name,
                )
            ax.set_title(task + " / fixed 24-question panel")
            ax.set_ylim(-0.01, ceiling)
            ax.set_ylabel("Accuracy (fraction)")
            ax.legend()
        fig.suptitle("Real data, SmolLM2-135M on M1 Pro — reduced-scale protocol")
        fig.tight_layout()
        fig.savefig(BASE / "stage-metrics.png", dpi=160)
        plt.close(fig)
        with (BASE / "REPORT.md").open("a") as stream:
            stream.write("\n![Measured real-task stage results](stage-metrics.png)\n")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        panels = [
            ("sft", "loss", "Supervised response NLL"),
            ("reward_model", "loss", "Pairwise reward-model loss"),
            ("dpo", "loss", "Normalized DPO loss"),
            ("rlvr", "policy_loss", "PPO policy loss / final inner epoch"),
            ("rlvr", "value_loss", "PPO value loss / final inner epoch"),
            ("rlvr", "mean_reward", "Sampled terminal reward / batch mean"),
        ]
        for name, summary in seeds.items():
            rows = [json.loads(line) for line in (BASE / name / "metrics.jsonl").read_text().splitlines()]
            for ax, (stage, key, title) in zip(axes.flat, panels):
                selected = [
                    r
                    for r in rows
                    if r["stage"] == stage and (stage != "rlvr" or r["epoch"] == summary["config"]["ppo_epochs"] - 1)
                ]
                ax.plot([r["step"] for r in selected], [r[key] for r in selected], label=name, alpha=0.8)
                ax.set_title(title)
                ax.set_xlabel("Stage optimizer / rollout step")
        axes[0, 0].legend()
        fig.suptitle("Recorded optimization traces — different objectives, not comparable accuracy scores")
        fig.tight_layout()
        fig.savefig(BASE / "training-traces.png", dpi=150)
        plt.close(fig)
        with (BASE / "REPORT.md").open("a") as stream:
            stream.write("\n![Recorded optimization and reward traces](training-traces.png)\n")
    print(f"Reported {len(seeds)} seeds, {len(branches)} branches, {len(benchmarks)} complete benchmark results.")


if __name__ == "__main__":
    main()
