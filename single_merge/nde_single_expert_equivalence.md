# NDE with $N=1$ Expert: Equivalence to `dare_single.py`

This document proves that the NDE (Normalize-Dropout-Erase) merge method,
when applied with a single RL expert ($N=1$), reduces to DARE-linear — the
exact method used by `single_merge/dare_single.py`.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\theta^{\text{SFT}}$ | Base (SFT+DPO) model parameters (one weight tensor) |
| $\theta^{\text{RL}}$ | Single RL expert's parameters |
| $\tau$ | Task vector: $\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}$ |
| $d$ | Number of elements in the tensor (`numel()`) |
| $p$ | Dropout keep-probability (the `density` parameter) |
| $w$ | Contribution weight |
| $j$ | Index over individual parameter elements, $j \in \{1, \dots, d\}$ |

All operations are **per weight tensor** — both NDE and dare_linear process
each named tensor independently.

---

## Part 1: What NDE Computes with $N=1$

We trace through each NDE step (from `longcat_merge/nde.py`) with a single
expert and show which steps collapse to identity.

### Step 1: Task Vector

$$
\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}
$$

```python
# longcat_merge/nde.py:37
task_vectors = [t.to(base_tensor.dtype) - base_tensor for t in tensors]
# With N=1: task_vectors = [tau]
```

Identical in both methods. In dare_linear (`generalized_task_arithmetic.py:217`):

```python
delta = x - base   # same computation
```

### Step 2: Normalize — IDENTITY when $N=1$

Normalization scales all task vectors to a common L2 norm. With one expert,
the target norm equals its own norm:

$$
\bar{n} = \frac{1}{N} \sum_{i=1}^{N} \|\tau_i\|_2 = \frac{1}{1} \|\tau\|_2 = \|\tau\|_2
$$

The rescaling factor becomes:

$$
\hat{\tau} = \tau \cdot \frac{\bar{n}}{\|\tau\|_2} = \tau \cdot \frac{\|\tau\|_2}{\|\tau\|_2} = \tau \cdot 1 = \tau
$$

```python
# longcat_merge/nde.py:40-44
norms = [tv.float().norm() for tv in task_vectors]   # norms = [||tau||_2]
mean_norm = sum(norms) / len(norms)                  # mean_norm = ||tau||_2
for i, norm in enumerate(norms):
    if norm > 1e-8:
        task_vectors[i] = task_vectors[i] * (mean_norm / norm)
        #                                   ^^^^^^^^^^^^^^^^
        #                                   = ||tau||_2 / ||tau||_2 = 1.0
```

**Result**: $\hat{\tau} = \tau$. Normalization is a no-op. $\square$

**Intuition**: Normalization exists to equalize magnitudes across *multiple*
domain experts so no single domain dominates. With one expert, there is
nothing to equalize.

### Step 3: Dropout — ACTIVE (same as dare_linear)

Bernoulli dropout with L1 rescaling is applied regardless of $N$:

$$
m_j \sim \text{Bernoulli}(p), \quad j = 1, \dots, d
$$

$$
\tilde{\tau}^{\text{masked}} = \tau \odot m
$$

$$
\tilde{\tau} = \tilde{\tau}^{\text{masked}} \cdot \frac{\|\tau\|_1}{\|\tilde{\tau}^{\text{masked}}\|_1}
$$

```python
# longcat_merge/nde.py:47-51
for i, d in enumerate(density):
    if d < 1.0:
        task_vectors[i] = bernoulli(
            task_vectors[i], density=d, rescale_norm=RescaleNorm.l1
        )
```

This calls `mergekit/sparsify.py:119-135`:

```python
# sparsify.py:131-134
mask = torch.bernoulli(
    torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
)
res = rescaled_masked_tensor(tensor.to(work_dtype), mask, rescale_norm)
```

Which applies the mask and L1-rescales via `sparsify.py:23-53`:

```python
# sparsify.py:37,40-42,53
masked = tensor * mask                        # tau * m
before_scale = tensor.abs().sum()             # ||tau||_1
after_scale = masked.abs().sum()              # ||tau * m||_1
return masked * (before_scale / after_scale)  # L1 rescaling
```

**dare_linear uses the exact same function.** In
`generalized_task_arithmetic.py:144-150`:

```python
tv_info["delta"] = sparsify(
    tv_info["delta"],
    density=tv_info["density"],
    method=self.method.sparsification_method,   # SparsificationMethod.random
    rescale_norm=self.rescale_norm,              # RescaleNorm.l1 (default_rescale=True)
)
```

Where `SparsificationMethod.random` dispatches to the same `bernoulli()`
function (`sparsify.py:119`).

**Result**: Both NDE and dare_linear compute
$\tilde{\tau} = \text{L1Rescale}(\tau \odot m)$ using identical code. $\square$

### Step 4: Erase — IDENTITY when $N=1$

The sign consensus erase mask computes, for each element $j$:

$$
\sigma_j = \text{sign}\!\left(\sum_{i=1}^{N} \tilde{\tau}_{ij}\right)
$$

$$
E_{ij} = \mathbb{1}\!\big[\text{sign}(\tilde{\tau}_{ij}) = \sigma_j\big]
$$

With $N=1$, the sum has a single term:

$$
\sigma_j = \text{sign}\!\left(\tilde{\tau}_{1j}\right)
$$

So the equality check becomes:

$$
E_{1j} = \mathbb{1}\!\big[\text{sign}(\tilde{\tau}_{1j}) = \text{sign}(\tilde{\tau}_{1j})\big] = \mathbb{1}[\text{True}] = 1
$$

for every element where $\tilde{\tau}_{1j} \neq 0$.

```python
# generalized_task_arithmetic.py:243-247
sign = delta.sign().to(mask_dtype)
sign_weight = delta.sum(dim=0)                          # = tau_tilde (N=1, sum over 1 element)
majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1  # = sign(tau_tilde)

# generalized_task_arithmetic.py:254
return sign == majority_sign   # sign(tau_tilde) == sign(tau_tilde) => True everywhere non-zero
```

**Exception**: Where $\tilde{\tau}_{1j} = 0$, `sign()` returns 0. Then
`majority_sign` is $+1$ (since $0 \geq 0$), so the check becomes $0 = +1$
which is `False`, giving $E_{1j} = 0$. But since $\tilde{\tau}_{1j} = 0$
already, masking it out changes nothing: $0 \cdot E_{1j} = 0$ regardless.

**Result**: The erase mask is $E = 1$ on all non-zero elements. It has no
effect on the output. $\square$

**Intuition**: Sign consensus resolves *conflicts between experts*. With one
expert, every element trivially agrees with itself — there are no conflicts
to resolve.

**dare_linear skips this step entirely**: its `consensus_method=None`
(`registry.py:57`), so `generalized_task_arithmetic.py:163` takes the
`else` branch, bypassing `get_mask()`.

### Step 5: Weighted Combination — Weight Cancels when $N=1$

NDE performs per-element weighted averaging:

$$
\Delta^{\text{merged}}_j = \frac{\sum_{i=1}^{N} w_i \cdot E_{ij} \cdot \tilde{\tau}_{ij}}{\sum_{i=1}^{N} w_i \cdot E_{ij}}
$$

With $N=1$ and $E_{1j} = 1$:

$$
\Delta^{\text{merged}}_j = \frac{w \cdot 1 \cdot \tilde{\tau}_j}{w \cdot 1} = \frac{w \cdot \tilde{\tau}_j}{w} = \tilde{\tau}_j
$$

```python
# longcat_merge/nde.py:58-65
weights = torch.tensor(weight, ...)        # [w]
erased_weights = weights * erase_mask      # [w * 1] = [w]
merged_delta = (stacked * erased_weights).sum(dim=0)  # w * tau_tilde
divisor = erased_weights.sum(dim=0).clamp(min=1e-6)   # w
merged_delta = merged_delta / divisor                  # w * tau_tilde / w = tau_tilde
```

**The weight $w$ cancels completely.** The NDE result is:

$$
\boxed{\theta^{\text{NDE}}_{N=1} = \theta^{\text{SFT}} + \tilde{\tau} = \theta^{\text{SFT}} + \text{L1Rescale}(\tau \odot m)}
$$

---

## Part 2: What `dare_single.py` Computes

`dare_single.py` uses `merge_method="dare_linear"`, which is registered as
(`registry.py:56-64`):

```python
GeneralizedTaskArithmeticMerge(
    consensus_method=None,              # no sign consensus (no erase step)
    sparsification_method=SparsificationMethod.random,  # bernoulli dropout
    default_normalize=False,            # don't divide by weight sum
    default_rescale=True,               # L1 rescaling after dropout
)
```

Tracing through `GTATask.execute()` (`generalized_task_arithmetic.py:119-184`):

**Step A — Task vector** (`line 217`):
$$
\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}
$$

**Step B — Sparsify** (`lines 134-150`): Bernoulli dropout + L1 rescale
$$
\tilde{\tau} = \text{L1Rescale}(\tau \odot m), \quad m_j \sim \text{Bernoulli}(p)
$$

**Step C — Weight and sum** (`lines 152-160`):
$$
\Delta^{\text{weighted}} = w \cdot \tilde{\tau}
$$

```python
# generalized_task_arithmetic.py:152-160
deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)  # shape: (1, ...)
weights = torch.tensor([tv["weight"] for tv in tvs], ...)  # [w]
weighted_deltas = deltas * weights                          # w * tau_tilde
```

**Step D — No consensus** (`lines 163-176`): `consensus_method=None`, so:
```python
# generalized_task_arithmetic.py:173-176
mixed_delta = weighted_deltas.sum(dim=0)   # = w * tau_tilde
divisor = weights.sum(dim=0)               # = w
divisor[divisor.abs() < 1e-8] = 1          # guard (w > 0, no-op)
```

**Step E — No normalization** (`line 178-179`): `normalize=False`, so
the division `mixed_delta /= divisor` is **skipped**:
```python
# generalized_task_arithmetic.py:178-179
if self.normalize:       # False for dare_linear
    mixed_delta /= divisor   # SKIPPED
```

**Step F — Return** (`line 184`):
$$
\boxed{\theta^{\text{dare\_linear}} = \theta^{\text{SFT}} + w \cdot \tilde{\tau}}
$$

---

## Part 3: Comparison

| | NDE ($N=1$) | dare_linear (`dare_single.py`) |
|---|---|---|
| Task vector | $\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}$ | $\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}$ |
| Dropout | $\tilde{\tau} = \text{L1Rescale}(\tau \odot m)$ | $\tilde{\tau} = \text{L1Rescale}(\tau \odot m)$ |
| Normalize step | Identity (mean norm = self norm) | N/A (no normalize step) |
| Erase step | Identity (sign agrees with self) | N/A (`consensus_method=None`) |
| Weight application | **Cancels**: $w\tilde{\tau}/w = \tilde{\tau}$ | **Preserved**: $w \cdot \tilde{\tau}$ |
| **Final result** | $\theta^{\text{SFT}} + \tilde{\tau}$ | $\theta^{\text{SFT}} + w \cdot \tilde{\tau}$ |

### The Only Difference: Weight Scaling

In NDE with $N=1$, the per-element normalization (dividing by weight sum)
causes $w$ to cancel:

$$
\frac{w \cdot \tilde{\tau}}{w} = \tilde{\tau}
$$

In dare_linear with `normalize=False`, no such division occurs, so the
weight $w$ directly scales the merged delta:

$$
w \cdot \tilde{\tau}
$$

**Exact equivalence holds when $w = 1.0$:**

$$
\theta^{\text{NDE}}_{N=1} = \theta^{\text{SFT}} + \tilde{\tau} = \theta^{\text{SFT}} + 1.0 \cdot \tilde{\tau} = \theta^{\text{dare\_linear}}_{w=1.0}
$$

### Why `dare_single.py` Uses $w < 1.0$

With a single expert, NDE's weight parameter is redundant (it cancels). But
in `dare_single.py`, the weight serves as a **task vector scaling factor**:

$$
\theta^{\text{merged}} = \theta^{\text{SFT}} + w \cdot \text{L1Rescale}(\tau \odot m)
$$

This is actually an advantage for the single-expert case. Setting $w = 0.7$
means: "apply 70% of the (dropout-pruned, rescaled) RL delta." This provides
a direct knob to control the strength of skill transfer:

- $w = 1.0$: Full RL delta (= NDE with $N=1$)
- $w = 0.7$: Conservative blend — preserves more of base model's behavior
- $w = 0.5$: Half-strength — maximal preservation of base capabilities

In the multi-expert case ($N > 1$), NDE's per-element normalization serves
the same purpose: it prevents experts from stacking up and overwhelming the
base. With $N=1$, this normalization has nothing to balance, so the weight
becomes the only control — and dare_linear exposes it directly.

---

## Part 4: Full Derivation — NDE Collapses Step by Step

Starting from the general NDE formula (from `longcat_merge/algo_explanation.md`):

$$
\theta^{\text{merged}}_j = \theta^{\text{SFT}}_j + \frac{
  \displaystyle\sum_{i=1}^{N} w_i \cdot E_{ij} \cdot \tilde{\tau}_{ij}
}{
  \displaystyle\sum_{i=1}^{N} w_i \cdot E_{ij}
}
$$

**Set $N = 1$:**

$$
\theta^{\text{merged}}_j = \theta^{\text{SFT}}_j + \frac{w_1 \cdot E_{1j} \cdot \tilde{\tau}_{1j}}{w_1 \cdot E_{1j}}
$$

**Substitute $E_{1j} = 1$** (proven in Part 1, Step 4):

$$
\theta^{\text{merged}}_j = \theta^{\text{SFT}}_j + \frac{w_1 \cdot 1 \cdot \tilde{\tau}_{1j}}{w_1 \cdot 1} = \theta^{\text{SFT}}_j + \tilde{\tau}_{1j}
$$

**Expand $\tilde{\tau}_{1j}$** — substituting normalization (identity) and
dropout:

$$
\tilde{\tau}_{1j} = \text{L1Rescale}\!\big(\hat{\tau}_{1j} \cdot m_j\big) = \text{L1Rescale}\!\big(\tau_{1j} \cdot m_j\big)
$$

where $\hat{\tau}_1 = \tau_1$ since normalization is identity.

**Therefore:**

$$
\theta^{\text{NDE}}_{N=1,\,j} = \theta^{\text{SFT}}_j + \text{L1Rescale}\!\big((\theta^{\text{RL}}_j - \theta^{\text{SFT}}_j) \cdot m_j\big)
$$

Compare with dare_linear at $w = 1$:

$$
\theta^{\text{dare\_linear}}_{w=1,\,j} = \theta^{\text{SFT}}_j + 1 \cdot \text{L1Rescale}\!\big((\theta^{\text{RL}}_j - \theta^{\text{SFT}}_j) \cdot m_j\big)
$$

$$
\boxed{\theta^{\text{NDE}}_{N=1} \equiv \theta^{\text{dare\_linear}}_{w=1}}
$$

And more generally:

$$
\theta^{\text{dare\_linear}} = \theta^{\text{SFT}} + w \cdot \text{L1Rescale}(\tau \odot m)
$$

is a strictly more expressive formulation for the single-expert case, since
$w$ provides a scaling degree of freedom that NDE's per-element normalization
eliminates.

---

## Summary

| NDE Component | With $N=1$ | Equivalent dare_linear Component |
|---|---|---|
| Task vector ($\tau = \theta^{\text{RL}} - \theta^{\text{SFT}}$) | Active | `delta = x - base` (`GTA:217`) |
| Normalize (scale to mean L2 norm) | **Identity** ($\bar{n}/n = 1$) | Not present |
| Dropout (Bernoulli + L1 rescale) | Active | `sparsify(..., method=random, rescale=l1)` (`GTA:144`) |
| Erase (sign consensus mask) | **Identity** ($E = 1$) | Not present (`consensus=None`) |
| Per-element weight normalization | **Cancels $w$** | Not present (`normalize=False`) |
| Weight scaling | No effect | $w \cdot \tilde{\tau}$ — controls merge strength |

**Bottom line**: With a single RL expert, two of NDE's three prongs (Normalize
and Erase) become identity operations and the third (Dropout) is exactly
dare_linear's sparsification. The only difference is that `dare_single.py`
preserves the weight as a scaling factor, giving you direct control over how
much RL delta to apply — a useful knob when tuning the math-vs-general
capability tradeoff.

---

## Code References

| What | File | Lines |
|------|------|-------|
| NDE merge function | `longcat_merge/nde.py` | 24-67 |
| dare_linear registration | `mergekit/merge_methods/registry.py` | 56-64 |
| `GTATask.execute()` | `mergekit/merge_methods/generalized_task_arithmetic.py` | 119-184 |
| `get_task_vectors()` | `mergekit/merge_methods/generalized_task_arithmetic.py` | 190-227 |
| `get_mask()` (sign consensus) | `mergekit/merge_methods/generalized_task_arithmetic.py` | 230-254 |
| `bernoulli()` dropout | `mergekit/sparsify.py` | 119-135 |
| `rescaled_masked_tensor()` | `mergekit/sparsify.py` | 23-53 |
| `dare_single.py` | `single_merge/dare_single.py` | 1-79 |
