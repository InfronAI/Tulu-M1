# Apple Silicon post-training guide

[English README](https://github.com/allenai/open-instruct/blob/main/README.md) · [中文 README](https://github.com/allenai/open-instruct/blob/main/README.zh-CN.md) · [Measured results](https://github.com/allenai/open-instruct/blob/main/experiments/apple_silicon/REPORT.md)

When reading an unpublished local checkout, use the root README files and `experiments/apple_silicon/REPORT.md`; the repository links resolve after these changes are published.

## Real-data extension

The [real-data study](tulu3_mac_experiments.md) supersedes the original fixture-only limitations for data adapters, math/IFEval grading, generation caching, and experiment design. It uses its own recorded runtime and report. The initial synthetic measurements below remain historical, with their original sources archived under `experiments/apple_silicon/source-v1`.

## Scope and design

The portable backend provides an inspectable single-process implementation of the Tülu 3 stage sequence. It is additive: the existing `finetune.py`, `dpo.py`, Core/vLLM, DeepSpeed/vLLM and MILES paths retain their existing behavior and dependencies. It does not claim that every original trainer now runs on macOS.

The chosen reference is [Tülu 3, arXiv:2411.15124v2](https://arxiv.org/html/2411.15124v2), especially sections 4–7. Tülu 3 uses normalized DPO and PPO-based RLVR. GRPO is an existing alternative in this repository, but is not substituted for PPO in this local reproduction of the stage sequence.

## Installation and device selection

```bash
PYTHON_BIN=python3.11 bash scripts/portable/setup.sh
.venv-mac/bin/python -m open_instruct.portable doctor
```

Requirements: native arm64 Python 3.11/3.12, macOS 14+ for this pinned dependency set, and free disk space for dependencies and checkpoints. Tested hardware is **Apple M1 Pro, 16 GB**, not a base M1 with 8 GB. CPU execution also works; only this Mac's CPU was measured. `requirements-mac.txt` deliberately avoids the root package's CUDA dependency graph. Do not run root `uv sync` to install the portable backend. The isolated runtime pins PyTorch 2.15 and Transformers 4.55 (the archived synthetic measurements used PyTorch 2.8), independently of the newer cluster stack.

`--device auto` selects MPS if available and CPU otherwise; `--device mps` fails explicitly if MPS is unavailable. The doctor performs real backward operations, not only an availability check. Unsupported MPS operations are not silently routed to CPU; use explicit `--device cpu` when necessary. Consult [PyTorch's MPS documentation](https://docs.pytorch.org/docs/2.8/notes/mps.html).

## Execute the entire local pipeline

```bash
.venv-mac/bin/python -m open_instruct.portable run \
  --config configs/portable/tiny.json --device mps --output output/my-first-run
```

The default is fully offline after installation. It creates a randomly initialized 124,864-parameter GPT-2 causal Transformer with a reversible byte tokenizer. Synthetic copy/addition data allow real gradients and deterministic reward verification without downloading a model or dataset. They are an integration fixture, not a replacement for a pretrained language model.

Stages execute in order:

1. **Prepare and validate:** five disjoint splits; schema, chosen/rejected validity, exact normalized-prompt overlap and context-length checks. Data generation has its own fixed seed (42).
2. **Base evaluation:** save the initial policy and evaluate validation/test prompts.
3. **SFT:** optimize next-token cross entropy on assistant responses, including EOS; prompt and padding tokens contribute no loss. Microbatches accumulate into one effective batch, normalized by the total response-token count.
4. **Reward model:** copy the SFT backbone, add a scalar head and train pairwise logistic loss at response EOS. This model initializes the PPO critic; its scores do not replace the verifiable reward.
5. **DPO-norm:** cache frozen SFT chosen/rejected response log probabilities, then optimize their length-normalized preference margin. Chosen and rejected forward passes are separate. Cached reference values are never recomputed from the updating policy.
6. **PPO-RLVR:** freeze a copy of the DPO policy on CPU, sample responses at temperature 1, verify them, and train policy/critic with token-level KL shaping, terminal rewards, GAE, clipped policy and value objectives. Reward is 10 for a correct EOS-terminated answer, 0 for an incorrect one, and -10 for missing EOS. Dropout is disabled. The generation cap is treated as terminal. The critic is trained separately from the policy.
7. **Evaluate and save:** greedy generation, exact match, chosen/rejected likelihood accuracy and response NLL after each policy stage. Save all predictions, including failures. Final checkpoints are fixed by step count; test results are not used for checkpoint selection.

No test/validation rows participate in optimization. Reported preference accuracy compares length-normalized chosen/rejected response likelihoods, not an LLM judge. NLL is the mean of per-example response NLLs. Training samples with replacement for a fixed number of steps and uses constant learning rates; these are explicit deviations from the paper's epoch-based mixture training and schedules.

## Pretrained model path

```bash
.venv-mac/bin/python -m open_instruct.portable run \
  --config configs/portable/smollm2.json --device mps --output output/smollm2
```

This downloads [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), pinned to revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It performs full-parameter FP32 updates, using the same synthetic dataset unless `--data` is supplied. This is a pretrained backend smoke experiment; its short training budget is not a recipe for a general-purpose assistant. The original model uses Apache 2.0 licensing. Allow several GB for the four policy checkpoints plus reward/value states and the download cache.

A Hugging Face model directory can also be supplied through the JSON `model` field. The backend uses `trust_remote_code=False`; it supports causal models exposing a standard `base_model`, `hidden_size` and `max_position_embeddings`. A model-specific chat template is used when present, otherwise the Tülu-style user/assistant delimiters are rendered as text. Architecture compatibility beyond GPT-2 and SmolLM2 is not asserted. This path is not an Olmo MoE port; use the existing MILES workflow for supported MoE models.

## Custom datasets

Pass `--data path/to/dataset.json`. The top-level object contains **exactly** `sft`, `preference`, `rlvr`, `validation`, `test`, each a nonempty array. Legacy synthetic rows use this explicit common schema:

```json
{
  "prompt": "Add 2 and 3. Answer with only the number.",
  "chosen": "5",
  "rejected": "6",
  "answer": "5",
  "task": "exact"
}
```

For the synthetic fixture, `exact` compares the entire stripped completion to `answer`; `integer` requires a final `#### 5` or `Answer: 5` marker. Chosen must verify and rejected must fail. For real data, SFT needs prompt/chosen, preference data need prompt/chosen/rejected, and RLVR/evaluation need prompt/task plus answer or constraints. GSM8K, symbolic MATH and strict/loose IFEval verifiers are provided by the real-data extension. Missing preference pairs do not receive invented likelihood scores. A prompt must not occur twice anywhere after case/whitespace normalization.

This narrow schema intentionally fails on raw Tülu mixtures: convert and validate them explicitly before use, preserving provenance and licenses. The [real-data extension](tulu3_mac_experiments.md) now provides stage-specific schemas, multi-turn context, original preference labels, symbolic math and IFEval adapters. Full parity with every original data/training/evaluation recipe is still not established. Original repository pipelines remain available for those workloads. No raw third-party mixture is silently relabeled as this fixture.

To inspect or modify the generated fixture:

```bash
.venv-mac/bin/python -m open_instruct.portable prepare --output output/fixture.json
```

This refuses to overwrite an existing file. Oversized examples fail before optimization, rather than truncating the supervised answer. Increase `max_length` and check memory, or shorten the input yourself.

## Checkpoints, artifacts and reproducibility

Each fresh output directory contains:

```text
config.json               resolved training configuration
dataset.json              exact data used in every split
metrics.jsonl             every optimization step / PPO epoch
rollouts.jsonl            sampled completions, token IDs, verifier and reward
dpo_reference.json        immutable reference scores and preference-data hash
evaluations.json          stage metrics and every validation/test prediction
summary.json              completion marker, environment/source hashes, timing, RSS
checkpoints/
  base/ sft/ dpo/ rlvr/    model config, weights and tokenizer
  reward_model.pt         critic initialization checkpoint
  value.pt                final critic weights
```

`dpo_skipped/` replaces `dpo/` in the no-DPO ablation. `summary.json` is written only after successful completion; interrupted directories remain available for diagnosis. Output directories must be fresh. The backend does **not** resume optimizer state mid-run. Inference reloads saved weights with `weights_only=True`; no arbitrary remote model code is enabled.

```bash
.venv-mac/bin/python -m open_instruct.portable generate \
  --checkpoint output/my-first-run/checkpoints/rlvr \
  --prompt "Copy 42. Answer with only the number." --max-new-tokens 5
```

Generate the validation matrix and report:

```bash
.venv-mac/bin/python scripts/portable/experiments.py \
  --include-pretrained --output output/my-experiments
.venv-mac/bin/python scripts/portable/report.py \
  --input output/my-experiments --demo-dir output/my-demo
```

The matrix runs three MPS seeds, a CPU control, four diagnostic ablations, and optionally the pinned 135M model, sequentially. It reuses completed runs only when their configuration and source hashes match; it does not resume incomplete runs. Seeds do not guarantee identical results across different hardware or PyTorch versions. The saved environment lock describes the measured Mac. Checkpoints stay on local disk and are git-ignored; raw measurements, configurations, predictions and analysis are retained in the project.

The demo renderer requires `ffmpeg` (`brew install ffmpeg`) in addition to Python dependencies. It renders the actual timestamped console recording and fixed first-four test predictions into a time-compressed replay. It is not a live screen capture. GIF loops in README renderers; the standalone HTML uses `autoplay muted loop playsinline controls` for MP4, subject to browser policy. [GitHub's markup pipeline](https://github.com/github/markup) sanitizes HTML, so README MP4 autoplay cannot be guaranteed.

## Memory and performance choices

- FP32 and PyTorch SDPA attention avoid CUDA-only dependencies. Historical synthetic results used eager attention; current CPU/MPS tests validate SDPA. Both devices use the same mathematical objectives.
- Training uses non-reentrant activation checkpointing. MPS allocation is capped at 55% of the recommended working set. Generation projects only the last hidden position to vocabulary logits; cached evaluation releases unused allocations per batch, and optimizer steps release unused MPS cache. CPU construction avoids transient accelerator model copies; the critic stays on CPU during DPO. These bounds do not guarantee that every model/context fits an 8 GB Mac.
- Tiny defaults: 128 context tokens, 5 generated tokens, effective batch 4, microbatch 2. The 135M preset uses microbatch 1 and a bounded generation budget.
- DPO caches only scalar sequence scores rather than keeping a second full reference model. The cache includes data identity and normalization settings.
- RL reference inference uses CPU in microbatches. On Apple Silicon, CPU and GPU share physical RAM; this does not make reference weights free. Policy/critic optimizer states dominate larger runs.
- Rollouts are sequential; generation now uses a KV cache, and evaluation batches left-padded prompts with explicit positions. CPU/MPS tests compare cached and batched output with uncached output. No claim of production-scale inference speed is made.
- Existing checkpoints are saved after every stage, enabling independent inference and comparison. Disk footprint is greater for the 135M path.

If memory pressure is high, start with `tiny`, reduce `micro_batch_size` and context/generation length, and close other memory-heavy applications. `batch_size` must be a multiple of `micro_batch_size`. Do not disable the MPS memory limit to conceal an out-of-memory failure. Timing and RSS from the actual machine are in the report.

## Paper coverage

This is an implementation and reproduction audit, **not a claim of 100% paper coverage**. The repository is a multi-generation research codebase; historical commands are not all runnable at current HEAD. In particular, `docs/tulu3.md` records removed legacy PPO entrypoints. The portable implementation restores a bounded local PPO-RLVR path; it does not restore distributed historical infrastructure or all experimental recipes.

| Paper component | Existing repository | Added Mac capability / evidence | Remaining gap |
|---|---|---|---|
| Data curation and decontamination (§3) | Data scripts and `decontamination/` | Pinned real-data subsets, split validation, exact-prompt and 13-word overlap screening | Full data synthesis, original mixture proportions and comprehensive contamination studies |
| SFT (§4) | `finetune.py`, OLMo-core workflows | Real response-masked causal LM updates on CPU/MPS | Original Llama 8B/70B/405B, original mixtures, schedules and all SFT ablations |
| Preference tuning (§5) | `dpo.py`, `dpo_utils.py` | Length-normalized DPO, frozen cached scores, summed-DPO diagnostic | LLM judges, on-policy preference collection and complete preference-mixture studies |
| Reward model / value initialization (§6) | `reward_modeling.py` | Pairwise RM training on published labels and critic initialization | Full preference mixture, original RM checkpoints and tuning |
| RLVR (§6) | Current GRPO alternatives; legacy Tülu PPO removed | Actual PPO, GAE, clipped policy/value losses, KL penalty, EOS penalty | Full protocol/grading parity, long-response budgets and asynchronous/distributed execution (real-task adapters now exist) |
| Evaluation (§7) | OLMES integration and historical eval scripts | Stage-wise local metrics, real-task adapters and resumable benchmark predictions | All published development/unseen/safety benchmarks and judge/human evaluations |
| Scaling and ablations | Original configs and historical run links | Synthetic device comparisons; real-data three-seed protocol and ten anchored RL branches with recorded completion status | Paper-wide experiments, original model scaling and published scores |

A 16 GB M1 Pro cannot hold and fully train the original 8B policy, reference, critic and optimizer states, let alone 70B/405B. Smaller models, LoRA or quantization would change the experiment; they cannot justify claiming equivalent paper results. Completing the remaining rows needs full-scale data and model runs, additional benchmark integrations, licensed judges where applicable, and suitable accelerator resources. None of those missing experiments is recorded as completed.

## Validation

```bash
make mac-check
.venv-mac/bin/python -m open_instruct.portable doctor
```

The focused tests cover data leakage, invalid input, UTF-8 tokenization, response/padding/EOS masking, gradient direction, analytic GAE/PPO cases, microbatch equivalence, cached-reference consistency, actual CPU/MPS optimization, model parameter changes, checkpoint inference, and overwrite protection. CUDA training code is unchanged; the Beaker GPU suite has not been run as part of this Mac task. Root `make style` / `make quality` remain the cluster-development commands and require that environment.
