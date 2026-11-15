import json

from scripts.portable import prepare_real_data, report_real, reproduce_real


def test_original_preference_labels_are_preserved():
    prompt = [dict(role="user", content="Write a poem")]
    row = dict(
        id="pair",
        source="published",
        chosen=prompt + [dict(role="assistant", content="A poem")],
        rejected=prompt + [dict(role="assistant", content="Another poem")],
    )
    result = prepare_real_data.convert("preference", row)
    assert result["chosen"] == "A poem" and result["rejected"] == "Another poem"
    assert "answer" not in result and "task" not in result


def test_fresh_reproduction_does_not_skip_training_due_to_recorded_results(tmp_path):
    workspace = reproduce_real.prepare(tmp_path / "fresh")
    manifest = json.loads((workspace / "workspace-manifest.json").read_text())
    assert "open_instruct/portable/train.py" in manifest["files_sha256"]
    assert (workspace / "experiments/tulu3_real/data-v2/dataset.json").is_file()
    assert not list(workspace.glob("experiments/tulu3_real/**/summary.json"))
    assert not list(workspace.rglob("*.pt"))
    assert (workspace / "configs/portable/tulu3-real.json").is_file()


def test_legacy_rlvr_json_null_and_demonstration_removal():
    raw = dict(
        dataset="ifeval",
        messages=[dict(role="user", content="No commas")],
        ground_truth=json.dumps(dict(func_name="validate_no_commas", N=None)),
    )
    result = prepare_real_data.convert("rlvr", raw)
    assert result["constraints"]["N"] is None
    raw = dict(
        dataset="MATH",
        messages=[dict(role="user", content="Question: 1+1?\nAnswer:2\n\nQuestion: 2+3?\nAnswer:")],
        ground_truth="5",
    )
    result = prepare_real_data.convert("rlvr", raw)
    assert result["prompt"] == "2+3?" and result["answer"] == "5"


def test_paired_comparison_and_wilson_zero_successes():
    control = [dict(prompt="a", correct=False), dict(prompt="b", correct=True)]
    variant = [dict(prompt="a", correct=True), dict(prompt="b", correct=False)]
    result = report_real.paired(control, variant)
    assert result == dict(wins=1, losses=1, exact_mcnemar_p=1.0)
    low, high = report_real.wilson(0, 1319)
    assert abs(low) < 1e-12 and 0 < high < 0.003
