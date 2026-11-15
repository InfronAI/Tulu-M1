# Tülu 3 on Mac: real-data experiments

Paper reference: [Tülu 3, arXiv v5](https://arxiv.org/abs/2411.15124v5).

This extends the initial [synthetic integration experiment](../experiments/apple_silicon/REPORT.md). It uses published Tülu training examples, unmodified preference labels, the released RLVR answers/constraints, and real GSM8K, MATH-500 and IFEval evaluation questions. It remains a resource-adapted study using SmolLM2-135M, not a reproduction of the original Llama 8B/70B/405B results.

## Reproduce

```bash
PYTHON_BIN=python3.11 MAC_ENV=.venv-mac-real bash scripts/portable/setup.sh
.venv-mac-real/bin/python -m scripts.portable.reproduce_real \
  --workspace output/tulu3-reproduction
```

Use this fresh-workspace entrypoint for a new clone: recorded result JSON is versioned, but model checkpoints are not. It copies computational sources and pinned input snapshots into a new portable workspace, records copied-file hashes, and runs training, complete benchmarks, preference holdout, decoding sensitivity, subgroup analysis, report, audit and demo rendering there. It refuses to overwrite an existing directory. `--prepare-only` copies inputs without launching training. The recorded source and results in the checkout remain available for comparison.

For the original measured workspace **when its local checkpoints are present**, these individual commands reuse completed runs:

```bash
.venv-mac-real/bin/python -m scripts.portable.real_experiments suite
.venv-mac-real/bin/python -m scripts.portable.preference_holdout
.venv-mac-real/bin/python -m scripts.portable.decode_sensitivity
.venv-mac-real/bin/python -m scripts.portable.benchmark_analysis
.venv-mac-real/bin/python -m scripts.portable.report_real
.venv-mac-real/bin/python -m scripts.portable.audit_real
```

To independently rebuild input snapshots, run `python -m scripts.portable.prepare_real_data --output output/tulu3-data-new` in the installed environment. Preparation refuses to overwrite an existing directory. The suite verifies its frozen source/data/configuration plan before restarting. A completed training run is reused; an interrupted training directory needs a fresh name, since optimizer-state resumption is not implemented. Benchmark generation is resumable, with checkpoint, data-order and protocol checks. Reusing summary files without their weights is not supported; use the fresh-workspace entrypoint instead.

The real-data runtime pins PyTorch 2.15.0. The initial synthetic study used 2.8.0 and remains archived with its original environment and source hashes. A 2.8 MPS real-data pilot stopped on a nonfinite gradient; its log and independent diagnostic are retained, not relabeled as a successful experiment. Updating the runtime is not by itself proof of the numerical cause. Successful runs and numerical tests establish only the validated workloads on this machine.

## Data and provenance

All source revisions and file SHA-256 hashes are saved in `experiments/tulu3_real/data-v2/manifest.json`. Downloaded datasets are cached by Hugging Face; only bounded selected examples, the evaluation questions, dataset cards and receipts are kept in this project.

| Use | Released source | Local adaptation |
|---|---|---|
| SFT | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Scan at most 8 row groups × 256 rows from each of all six pinned shards; balance observed sources; select 96 length-eligible examples |
| DPO / reward model | [allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture) | Same bounded scan across all four shards; 48 original chosen/rejected pairs plus 16 unused holdout pairs; reject identical pairs |
| RLVR | [allenai/RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints) | 24 GSM8K, 24 MATH, 24 legacy IF prompts; retain answers and constraint parameters |
| Math evaluation | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Official test questions; no generated replacement questions |
| Instruction evaluation | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | All original instruction IDs and kwargs preserved |

The selected SFT sample covers 15 source labels and the preference sample covers all seven source labels. These small balanced samples **do not preserve the paper's mixture proportions**. Their selection is determined by source, seed and sequence length, never by model correctness. Some original mixture constituents have non-commercial conditions; consult the retained dataset cards and upstream component licenses. Do not treat this research collection as uniformly Apache-licensed data.

The training inputs are screened against all 2,360 evaluation prompts using normalized prompt equality and 13-word overlap, including training responses. This is a conservative text-overlap check, not proof against semantic or pretraining contamination. Original selected rows are retained for inspection. Earlier single-shard pilots and failed preparations are explicitly separate from `data-v2`.

## Training protocol

The model is pinned SmolLM2-135M in FP32. The context limit is 768 tokens, generation cap 128, effective batch 2 and microbatch 1. Training iterates shuffled epochs rather than sampling with replacement. It uses linear learning-rate decay with 10% warmup.

- SFT: 96 optimizer steps, approximately two passes through 96 examples, learning rate 5e-6. Earlier chat turns remain in the prompt; only the final assistant response receives supervised loss.
- Reward model: 48 steps on 48 preference pairs, scalar EOS head and pairwise logistic loss, learning rate 5e-7. It initializes the critic.
- DPO: 24 steps, one pass through 48 pairs, beta 5, response-length normalization and cached frozen SFT log probabilities, learning rate 5e-7.
- PPO-RLVR: 12 rollout steps × two prompts, four PPO updates per rollout, learning rate 3e-7. Reward scale 10, missing-EOS penalty -10, reference KL beta 0.05, GAE lambda 0.95 and clip 0.2. Only 24 of 72 RL prompts are consumed in a baseline run; this is a short bounded RL experiment.

Source and environment receipts are stored per run. The paper's full epoch counts, batch sizes, model pretraining, long contexts, value-model training and asynchronous infrastructure are not reproduced by these budgets.

## Verifier and prompt conventions

- GSM8K extracts the final signed numeric result, removes digit separators and compares with `Decimal`; it does not require a newly invented answer marker.
- MATH reuses `open_instruct.math_utils`: boxed/Minerva/LaTeX extraction, normalized equality and symbolic equivalence. Symbolic comparison has a five-second timeout; free-text candidate strings over 512 characters skip symbolic parsing and record that skip. This additional bound can affect parity.
- The released RLVR IF examples use the old `func_name` format. Training uses the repository's `if_functions` implementations with their actual parameters. Empty responses receive no correctness reward.
- Official IFEval questions use the repository's `IFEvalG` instruction registry. Evaluation reports strict and loose prompt-level and instruction-level scores. This registry contains upstream fixes; it is not claimed to be byte-identical to Google's original scorer. Checker randomness is isolated from training RNG.

For constrained local training, GSM8K/MATH demonstrations are removed from the released RLVR prompt, retaining the last target question and original answer. Evaluation uses zero-shot prompts, not the paper's few-shot OLMES protocol. A 128-token generation cap is often insufficient for mathematical reasoning or long IFEval responses; EOS/truncation rates are reported. Full question coverage under this cap is **not full protocol fidelity**. The primary table uses question-level micro accuracy and strict IFEval. Supplementary analysis also computes MATH-500 subject macro accuracy and IFEval loose prompt accuracy, matching the aggregation families used in paper §7.2 while retaining the other protocol differences.

Generation uses a KV cache and computes vocabulary logits only for the final position. Batched evaluation uses left padding and explicit position IDs. SDPA attention, non-reentrant activation checkpointing during training, a 55% MPS working-set allocation cap, cache release after optimizer/evaluation steps, and CPU construction of reference copies bound memory use. The critic is offloaded during DPO. An earlier eager-attention attempt reached about 12 GB physical footprint and severe swapping; it was stopped and archived under `attempt-memory-main-seed1`. A capped SDPA attempt ran out of memory during RM and is retained under `attempt-cap-main-seed1`; activation checkpointing was added after that failure. Tests compare cached/batched greedy output with uncached single-example output on CPU and MPS. Rollout sampling remains temperature 1, matching the distribution whose probabilities enter PPO.

## Comparisons

Three complete training runs vary the seed. Ablations fork from byte-identical saved seed-1 SFT/DPO/RM checkpoints; every fork starts fresh optimizer state and a common RL sampling seed. A separately rerun control provides the correct comparison for these forks. The frozen runner explicitly sets initialization seed 101, optional summed-DPO seed 102 and RL seed 103. Branch configuration files inherit the baseline `seed: 1`; the explicit runner resets determine the actual branch RNG, as recorded in `branch-rng-protocol.json`.

The planned forks cover KL = 0/0.01/0.05/0.1, skipping DPO, summed DPO with beta 0.1, a policy-initialized critic with a fresh scalar head, and training RLVR on GSM8K-only, MATH-only or IF-only prompts. Summed DPO changes both normalization and beta, so it is a recipe comparison rather than an isolated normalization effect. Task-specific runs change the prompt distribution. Equal RNG seeds cannot force identical tokens after policies diverge.

A supplementary decoding-budget check compares 128 and 512 generated tokens on the same 72-question panel for the seed-1 base and final checkpoints. It is not used to select a model or adjust training. Run it with `.venv-mac-real/bin/python -m scripts.portable.decode_sensitivity` after the suite.

Stage evaluation uses eight validation and 24 test questions per task. Hyperparameters and final-step checkpoints are fixed before observing those results. Final benchmark execution covers the full 1,319 GSM8K test questions, 500 MATH-500 questions and 541 IFEval prompts, with one persisted prediction per original row and explicit completion markers. Read the result report for actual completion status; a planned or interrupted run is never counted as complete.

## Remaining paper gaps

Even after these experiments, the study does not reproduce the original base models, full training mixtures, scaling to 8B/70B/405B, extensive data-synthesis/judge ablations, learned reward plus verifier sweeps, model soups, distributed throughput, or the entire OLMES development/unseen/safety suite. MATH-500 is a 500-question subset, not the entire 5,000-question MATH test set. Judge and human preference evaluations are not replaced with invented proxy scores.

The purpose is to maximize inspectable, real-data coverage on the available M1 Pro while keeping deviations explicit. Zero gains, failures, verifier timeouts and truncated responses belong in the results alongside successful runs.
