# NDE (Normalize-Dropout-Erase) Merge Method

## Algorithm

Given a base SFT model and N domain-expert RL models, for each weight tensor:

1. **Normalize** — Compute task vectors `tau_i = theta_RL_i - theta_SFT`, then
   scale each to the mean L2 norm across experts. This balances contributions so
   no single domain dominates due to larger parameter changes.

2. **Dropout** — Apply DARE-style Bernoulli dropout: randomly zero out elements
   with probability `1 - density`, then rescale surviving elements by their L1
   norm ratio to preserve expected magnitude.

3. **Erase** — Apply SCE-style sign consensus: for each parameter position,
   determine the majority sign direction (weighted by magnitude). Zero out any
   expert's contribution that disagrees with the majority.

4. **Combine** — Weighted sum with per-element normalization:
   `merged = sum(w_i * mask_i * tau_i) / sum(w_i * mask_i)`

5. **Output** — `theta_merged = theta_SFT + merged`

## What's Determined in Mergekit (Reused Directly)

| Component | Source file | Role |
|---|---|---|
| `@merge_method` decorator | `mergekit/merge_methods/easy_define.py` | Auto-registers the method into mergekit's registry. Handles parameter parsing, task graph construction, and execution scheduling. |
| `get_mask()` | `mergekit/merge_methods/generalized_task_arithmetic.py:230` | Sign consensus mask (the **Erase** step). Given stacked deltas, computes majority sign per element and returns a boolean mask. Uses the `"sum"` method where sign votes are weighted by magnitude. |
| `bernoulli()` | `mergekit/sparsify.py:119` | DARE-style random dropout (the **Dropout** step). Generates a Bernoulli mask with keep-probability = `density`, applies it, and optionally rescales to preserve the original L1 norm. |
| `RescaleNorm.l1` | `mergekit/sparsify.py:17` | L1 norm rescaling mode. After dropout, the surviving elements are scaled so their L1 norm matches the pre-dropout tensor. |
| `run_merge()` | `mergekit/merge.py` | Full merge pipeline: loads models, builds computation graph, executes tensor-by-tensor, writes output. |
| `MergeConfiguration` | `mergekit/config.py` | YAML config parsing with parameter resolution (global, per-model, gradients, filters). |
| CLI infrastructure | `mergekit/options.py`, `mergekit/scripts/run_yaml.py` | Click-based CLI with `--cuda`, `--dtype`, etc. |

## What's Determined in the Paper

- **Task vector definition**: `tau_i = theta_i_RL - theta_SFT` — the parameter
  delta between each RL-trained domain expert and the shared SFT base.

- **Three-step pipeline**: The specific combination of Normalization, Dropout,
  and Erasure as a unified strategy. Prior methods use subsets (DARE = dropout
  only; TIES/SCE = sparsify + erase; task arithmetic = nothing), but this paper
  combines all three.

- **Ordering**: Normalize first, then Dropout, then Erase. Normalization operates
  on full task vectors before any sparsification.

- **Normalization purpose**: "Normalize the magnitude of task vectors to balance
  contributions from different domains" — prevents one expert from dominating.

- **Dropout purpose**: "Similar to DARE, apply dropout to prune redundant delta
  parameters" — reduces parameter interference.

- **Erase purpose**: "Inspired by SCE, erase parameter elements with
  minority-direction updates" — resolves sign conflicts between experts.

## What's Inferred

- **Normalization target = mean L2 norm**: The paper says "normalize the
  magnitude" but does not specify the target norm. We normalize each task
  vector's L2 norm to the mean L2 norm across all experts. This preserves the
  overall parameter scale while equalizing relative contributions. (Alternative:
  unit normalization would lose scale information.)

- **Per-tensor normalization**: The merge processes each weight tensor
  independently (mergekit's `@merge_method` operates on one tensor at a time).
  Normalization happens within each tensor — e.g., the attention Q-projection
  task vectors are normalized among themselves, separately from the MLP
  task vectors.

- **L1 rescaling after dropout**: The paper says "similar to DARE" for dropout
  but does not specify the rescaling norm. We use L1 (matching DARE's default in
  mergekit) which scales surviving elements by `||original||_1 / ||masked||_1`.

- **Per-element normalization in combination**: After applying the erase mask,
  we divide by the sum of active weights at each element position. This follows
  the pattern used in both SCE (`sce.py:42`) and TIES
  (`generalized_task_arithmetic.py:178-179` with `normalize=True`).

- **Guard values**: `1e-8` epsilon for zero-norm task vectors (skip
  normalization if a model's delta is effectively zero); `1e-6` clamp on the
  divisor (prevents NaN where all experts are erased at a position).

- **`density` as per-model tensor parameter**: Allows different dropout rates
  per expert (e.g., math expert at 0.9, code expert at 0.85). The paper does
  not discuss per-model density, but the mergekit convention supports it
  naturally.

## Usage

```bash
# From the mergekit repo root:
python -m longcat_merge.run longcat_merge/example_config.yaml ./output

# With GPU acceleration:
python -m longcat_merge.run longcat_merge/example_config.yaml ./output --cuda
```

See `example_config.yaml` for the configuration format.
