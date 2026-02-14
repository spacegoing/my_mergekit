# DARE Single-Model Merge — Hyperparameter Recommendations

## Your Situation

| | Base (SFT+DPO) | RL Checkpoint |
|---|---|---|
| AIME | 60@1 | 83@1 |
| Entropy | 0.3 | 0.4 |
| KL reg | — | None |

The RL run gained +23 AIME points but increased entropy by 33%. Without KL
regularization, the task vector contains both **math skill** (what you want)
and **distribution drift** (what you don't want). The merge needs to be
surgical.

## What the Two Knobs Do

For dare_linear with a single RL model, the output is:

```
result = base + weight * L1Rescale(delta * Bernoulli(density))
```

**`weight`** — scales the entire task vector. Lower = less RL influence.
This is a blunt instrument: it scales math gains and entropy drift equally.

**`density`** — fraction of delta parameters kept (rest randomly zeroed).
After zeroing, surviving elements are L1-rescaled to preserve the original
magnitude. This is the DARE insight: models are highly parameter-redundant,
so you can drop many delta elements without losing the learned capability.

The key subtlety: **density does NOT reduce the magnitude**. L1 rescaling
compensates. Instead, density controls **how many parameters are modified**.
Lower density = fewer but individually larger parameter changes. The
randomness acts as regularization — different random seeds will keep
different subsets, but the model-level behavior is preserved because the
delta is redundant.

## Recommended: `weight=0.7, density=0.5`

```bash
python longcat_merge/dare_single.py \
  --base /path/to/sft_dpo_base \
  --rl /path/to/math_rl \
  --out /path/to/merged \
  --weight 0.7 --density 0.5
```

### Why weight=0.7

With KL regularization, the RL optimizer constrains the policy to stay
close to the base. Your RL had no such constraint, so the task vector
magnitude is inflated relative to what a KL-regularized run would produce.
The entropy increase (0.3 → 0.4, +33%) is a proxy for this excess drift.

Scaling by 0.7 shaves 30% off the task vector. This partially compensates
for the missing KL penalty. The math gains are large enough (+23 AIME
points) that even at 70% scaling, substantial improvement should transfer.

Evidence from the codebase — the della test uses `lambda=0.5` to scale
task vectors after pruning:

```python
# tests/test_basic_merges.py:267
params={"density": 0.66, "epsilon": 0.05, "lambda": 0.5},
```

And the ties.yml example uses weight=0.5 for WizardMath on MLP layers,
weight=0 on attention — showing that partial scaling is standard practice:

```yaml
# examples/ties.yml:13-16
  - model: WizardLM/WizardMath-13B-V1.0
    parameters:
      density: 0.33
      weight:
        - filter: mlp
          value: 0.5
        - value: 0
```

### Why density=0.5

DARE's core finding is that delta parameters are highly redundant. The
paper shows models can tolerate dropping 90%+ of delta parameters in many
settings. 50% is a moderate starting point that:

1. Removes roughly half the parameter changes, reducing the chance of
   destructive interference with base capabilities
2. Is conservative enough to preserve the math signal
3. After L1 rescaling, the surviving parameters carry the same total
   magnitude, so the math skill encoded in the weight direction is preserved

Evidence — the bio-merge example uses density=0.5 for domain-specific
merging:

```yaml
# examples/bio-merge.yml:4-5
  - model: mistralai/Mistral-7B-Instruct-v0.2
    parameters:
      density: 0.5
      weight: 0.5
```

And mergekit's test suite uses density=0.66 for dare_ties and density=0.3
for ties, showing the typical range is 0.3–0.7:

```python
# tests/test_basic_merges.py:225
params={"density": 0.66},      # dare_ties test

# tests/test_basic_merges.py:177
params={"density": 0.3},       # ties test
```

### Why this combo works for your case

The two knobs are complementary:
- **density=0.5** removes redundant parameters (noise reduction)
- **weight=0.7** scales the remainder (drift compensation)

The effective modification to each surviving parameter is
`weight / density_realized ≈ 0.7 / ~0.5 ≈ 1.4x` the original delta
element value (due to L1 rescaling), but spread across only 50% of
parameters. This concentrates the math signal into fewer, more impactful
changes while reducing the overall footprint of modifications to the base.

## Alternative 1 (conservative): `weight=0.5, density=0.3`

```bash
python longcat_merge/dare_single.py \
  --base ... --rl ... --out ... \
  --weight 0.5 --density 0.3
```

**When to try this**: If the recommended combo still damages other domains
too much (e.g., instruction following degrades, or coding benchmarks drop).

- **density=0.3**: Very aggressive pruning. Only 30% of delta parameters
  survive, but L1 rescaling keeps the magnitude. This is closer to DARE's
  finding that models tolerate extreme sparsification. Fewer parameters
  modified = smaller blast radius on base capabilities.

- **weight=0.5**: Halves the task vector. Combined with 30% density, this
  is the most conservative option. You might lose some AIME points (maybe
  70-75 instead of 80+), but base capabilities should be very well preserved.

Evidence — the ties test uses density=0.3 and it produces valid merges:

```python
# tests/test_basic_merges.py:176-177
merge_method="ties",
params={"density": 0.3},
```

## Alternative 2 (aggressive): `weight=1.0, density=0.7`

```bash
python longcat_merge/dare_single.py \
  --base ... --rl ... --out ... \
  --weight 1.0 --density 0.7
```

**When to try this**: If the recommended combo loses too much AIME
performance (e.g., drops from 83 to below 75).

- **density=0.7**: Mild pruning. 70% of parameters survive. Less noise
  removal, but also less risk of pruning something important.

- **weight=1.0**: Full task vector magnitude. No scaling down. This
  maximizes math transfer but does nothing to compensate for the entropy
  drift.

Evidence — the dare_ties test uses density=0.66, close to 0.7:

```python
# tests/test_basic_merges.py:223-225
merge_method="dare_ties",
params={"density": 0.66},
```

And the two_model_config helper uses weight=0.6 as a reasonable
single-model weight, suggesting 0.6–1.0 is the working range:

```python
# tests/test_basic_merges.py:306
parameters={"weight": 0.6},
```

## Summary Table

| Config | weight | density | Expected AIME | Base preservation | When |
|--------|--------|---------|---------------|-------------------|------|
| **Recommended** | 0.7 | 0.5 | ~78-80 | Good | Start here |
| Conservative | 0.5 | 0.3 | ~70-75 | Best | If base degrades |
| Aggressive | 1.0 | 0.7 | ~80-83 | Moderate | If math drops too much |

Run all three and eval on both AIME and your other-domain benchmarks to
find the Pareto-optimal point.

## How It Works in the Code

For a single RL model, dare_linear executes this path:

```python
# mergekit/merge_methods/generalized_task_arithmetic.py:124-184

# 1. Task vector
delta = rl_tensor - base_tensor                     # line 217

# 2. Bernoulli dropout + L1 rescale
delta = sparsify(delta, density, method=random,     # line 144-150
                 rescale_norm=RescaleNorm.l1)

# 3. Weight (N=1, so sum = single element)
weighted_delta = delta * weight                     # line 160

# 4. No consensus (consensus_method=None)
mixed_delta = weighted_delta.sum(dim=0)             # line 174
# (sum of 1 element = itself)

# 5. normalize=False, so no division                # line 178-179 skipped

# 6. lambda=1.0 by default, so no extra scaling     # line 181-182 skipped

# 7. Result
result = base + mixed_delta                         # line 184
```
