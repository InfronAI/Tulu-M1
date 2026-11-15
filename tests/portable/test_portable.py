import copy
import json
import math
import random
import sys

import pytest
import torch

from open_instruct.portable import data, model, objectives, train


def test_dataset_is_disjoint_and_reproducible():
    dataset = data.load()
    assert dataset == data.load()
    assert sum(map(len, dataset.values())) == 400
    dataset["test"][0]["prompt"] = "  " + dataset["sft"][0]["prompt"].upper() + "  "
    with pytest.raises(ValueError, match="contaminated"):
        data.validate(dataset)


@pytest.mark.parametrize("change", [{"chosen": "wrong"}, {"task": "unknown"}, {"answer": 2}, {"rejected": ""}])
def test_bad_data_fails_early(change):
    dataset = data.load()
    dataset["sft"][0].update(change)
    with pytest.raises(ValueError):
        data.validate(dataset)


def test_integer_verifier_does_not_reward_incidental_numbers():
    row = dict(task="integer", answer="42")
    assert data.verify(row, "Reasoning 17.\n#### 42")
    assert not data.verify(row, "42 was an intermediate result; answer is 19")
    assert not data.verify(row, "#### 42 and 19")


def test_utf8_mask_eos_and_overflow():
    tokenizer = data.ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("中文 café")) == "中文 café"
    row = data.load()["sft"][0]
    encoded = data.encode(tokenizer, row, "chosen", 128)
    tokens, attention, mask = model.batch([encoded, ([1, 8, 2], 1)], 0, "cpu")
    prefix = encoded[1]
    assert mask[0, : prefix - 1].sum() == 0
    assert mask[0].sum() == len(encoded[0]) - prefix
    assert mask[1].sum() == 2  # Includes EOS, excludes padding.
    assert attention[1].sum() == 3
    logits = torch.randn(*tokens.shape, 259, requires_grad=True)
    loss = -(objectives.token_logps(logits, tokens) * mask).sum()
    loss.backward()
    assert logits.grad[0, : prefix - 1].abs().sum() == 0
    assert logits.grad[1, 2:].abs().sum() == 0
    with pytest.raises(ValueError, match="No silent truncation"):
        data.encode(tokenizer, row, "chosen", 8)


def test_dpo_initial_loss_and_gradient_direction():
    chosen = torch.tensor([-2.0], requires_grad=True)
    rejected = torch.tensor([-3.0], requires_grad=True)
    loss = objectives.dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), 5)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()


def test_gae_terminal_and_padding():
    rewards = torch.tensor([[0.0, 10.0, 0.0], [5.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0, 999.0], [2.0, 999.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    advantages, returns = objectives.gae(rewards, values, mask, lam=1.0)
    torch.testing.assert_close(advantages, torch.tensor([[9.0, 8.0, 0.0], [3.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[10.0, 10.0, 0.0], [5.0, 0.0, 0.0]]))


def test_ppo_clips_positive_and_negative_advantages():
    new = torch.tensor([[math.log(2), math.log(0.5)]], requires_grad=True)
    zero = torch.zeros_like(new)
    policy, value = objectives.ppo_loss(new, zero, zero, zero, torch.tensor([[1.0, -1.0]]), zero, zero + 1)
    assert policy.item() == pytest.approx((-1.2 + 0.8) / 2)
    assert value.item() == 0
    policy.backward()
    torch.testing.assert_close(new.grad, zero)


def test_gradient_accumulation_matches_full_batch(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    lm, tokenizer = model.create("tiny", "main", 128)
    other = copy.deepcopy(lm)
    rows = data.load()["sft"][:5]
    cfg = train.Config(sft_steps=1, batch_size=4, micro_batch_size=4)
    random.seed(7)
    train.supervised(lm, tokenizer, rows, cfg, "cpu", train.Recorder(tmp_path / "full.jsonl"))
    cfg.micro_batch_size = 1
    random.seed(7)
    train.supervised(other, tokenizer, rows, cfg, "cpu", train.Recorder(tmp_path / "micro.jsonl"))
    for left, right in zip(lm.parameters(), other.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_activation_checkpointing_preserves_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(31)
    lm, tokenizer = model.create("tiny", "main", 128)
    reference = copy.deepcopy(lm)
    reference.gradient_checkpointing_disable()
    assert lm.is_gradient_checkpointing and not reference.is_gradient_checkpointing
    encoded = [data.encode(tokenizer, r, "chosen", 128) for r in data.load()["sft"][:2]]
    tokens, attention, mask = model.batch(encoded, tokenizer.pad_token_id, device)
    for policy in (lm, reference):
        policy.to(device).train()
        loss = -(model.logps(policy, tokens, attention) * mask).sum() / mask.sum()
        loss.backward()
    for actual, expected in zip(lm.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None and expected.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS")
        ),
    ],
)
def test_complete_pipeline_and_checkpoint_roundtrip(tmp_path, device):
    cfg = train.Config(
        device=device,
        sft_steps=2,
        rm_steps=2,
        dpo_steps=2,
        rlvr_steps=2,
        batch_size=2,
        micro_batch_size=1,
        ppo_epochs=2,
        max_new_tokens=3,
    )
    dataset = {key: rows[:2] for key, rows in data.load().items()}
    output = tmp_path / device
    result = train.run(cfg, dataset, output)
    assert set(result["evaluations"]) == {"base", "sft", "dpo", "rlvr"}
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert {r["stage"] for r in metrics} == {"sft", "reward_model", "dpo", "rlvr"}
    assert all(math.isfinite(r["loss"]) for r in metrics)
    before, tokenizer = model.restore(output / "checkpoints" / "sft", "cpu")
    after, _ = model.restore(output / "checkpoints" / "rlvr", "cpu")
    assert any(not torch.equal(a, b) for a, b in zip(before.parameters(), after.parameters()))
    dpo, _ = model.restore(output / "checkpoints" / "dpo", "cpu")
    assert any(not torch.equal(a, b) for a, b in zip(dpo.parameters(), after.parameters()))
    reward = torch.load(output / "checkpoints" / "reward_model.pt", weights_only=True)
    value = torch.load(output / "checkpoints" / "value.pt", weights_only=True)
    assert any(not torch.equal(reward[k], value[k]) for k in reward)
    # Cached reference probabilities must still equal the immutable SFT checkpoint.
    cache = json.loads((output / "dpo_reference.json").read_text())
    encoded = [data.encode(tokenizer, row, "chosen", cfg.max_length) for row in dataset["preference"]]
    with torch.no_grad():
        actual = model.sequence_logps(before, encoded, tokenizer.pad_token_id, "cpu")
    torch.testing.assert_close(actual, torch.tensor(cache["chosen"]), atol=2e-5, rtol=2e-5)
    saved_predictions = json.loads((output / "evaluations.json").read_text())["rlvr"]["test"]["predictions"]
    assert (
        model.generate(after, tokenizer, dataset["test"][0]["prompt"], 3, "cpu")[2]
        == saved_predictions[0]["completion"]
    )
    assert not {"vllm", "deepspeed", "ray"}.intersection(sys.modules)
    with pytest.raises(FileExistsError):
        train.run(cfg, dataset, output)


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 3}, {"kl_beta": -1}, {"sft_steps": 0}, {"learning_rate": float("nan")}]
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        train.Config(**kwargs).validate()


def test_real_schema_does_not_invent_correctness_labels():
    dataset = data.load()
    dataset["sft"] = [dict(prompt=[dict(role="user", content="Hello")], chosen="Hello there")]
    dataset["preference"] = [dict(prompt="Write an essay", chosen="A careful essay", rejected="No")]
    data.validate(dataset)


def test_real_verifiers_and_loose_ifeval():
    from_example = dict(task="gsm8k", answer="-1200")
    assert data.verify(from_example, "First 7; the result is -1,200.00")
    assert not data.verify(from_example, "The result is 1200")
    assert data.verify(dict(task="math", answer=r"\frac{1}{2}"), r"Thus \boxed{0.5}")
    assert not data.verify(dict(task="math", answer="8"), r"\boxed{9}")
    row = dict(
        task="ifeval",
        prompt="Say yes",
        constraints=dict(instruction_id=["change_case:english_lowercase"], kwargs=[{}]),
    )
    assert data.verify(row, "this is a simple english sentence that uses only lowercase letters.")
    assert not data.verify(row, "YES")
    assert not data.verify(row, "")
    legacy = dict(task="ifeval", prompt="No comma", constraints=dict(func_name="validate_no_commas"))
    assert data.verify(legacy, "yes")
    assert not data.verify(legacy, "yes, indeed")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")),
    ],
)
def test_cached_batched_generation_matches_full_forward(device):
    torch.manual_seed(9)
    lm, tokenizer = model.create("tiny", "main", 128)
    lm.to(device).eval()
    prompts = ["a", "a much longer user prompt"]
    batched = model.generate_batch(lm, tokenizer, prompts, 6, device)
    for prompt, result in zip(prompts, batched):
        assert result == model.generate(lm, tokenizer, prompt, 6, device, use_cache=False)
        assert result == model.generate(lm, tokenizer, prompt, 6, device, use_cache=True)


def test_epoch_sampler_visits_every_example_before_repeating():
    cfg = train.Config(batch_size=2, micro_batch_size=1, sampling="epochs")
    iterator = train.sampled_batches(list(range(6)), cfg)
    assert sorted(next(iterator) + next(iterator) + next(iterator)) == list(range(6))


def test_verification_preserves_training_rng():
    row = dict(
        task="ifeval", prompt="No commas", constraints=dict(instruction_id=["punctuation:no_comma"], kwargs=[{}])
    )
    random.seed(71)
    before = random.getstate()
    assert data.verify(row, "this is an answer")
    assert random.getstate() == before


def test_linear_warmup_schedule():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    cfg = train.Config(lr_schedule="linear", warmup_ratio=0.2)
    rates = [train.learning_rate(optimizer, 1.0, step, 10, cfg) for step in range(1, 11)]
    assert rates[:3] == [0.5, 1.0, 1.0]
    assert rates[-1] == 0.125
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))
