# NDE Merge — Detailed Algorithm Explanation

## Notation

| Symbol | Meaning |
|--------|---------|
| $\theta^{\text{SFT}}$ | SFT base model parameters (one weight tensor) |
| $\theta_i^{\text{RL}}$ | RL domain-expert model $i$'s parameters |
| $\tau_i$ | Task vector for expert $i$ |
| $N$ | Number of domain experts |
| $d$ | Total number of elements in the tensor (i.e. `numel()`) |
| $p_i$ | Dropout keep-probability for expert $i$ (the `density` parameter) |
| $w_i$ | Contribution weight for expert $i$ |
| $j$ | Index over individual parameter elements, $j \in \{1, \dots, d\}$ |

All operations below are **per weight tensor** — mergekit invokes the merge
function independently for each named tensor in the model (e.g.,
`model.layers.0.self_attn.q_proj.weight`). The decorator in
`mergekit/merge_methods/easy_define.py:142-166` separates the base tensor from
the expert tensors and calls our function once per tensor.

---

## Step 0: Input Dispatch (easy_define.py)

The `@merge_method` decorator generates a Pydantic `Task` whose `execute()`
receives a `Dict[ModelReference, torch.Tensor]` containing all loaded tensors
for a given parameter name.

```python
# mergekit/merge_methods/easy_define.py:142-152
def _execute(self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs):
    model_refs = set(tensors.keys())
    base_model = getattr(self, "base_model", None)
    if base_model and base_model in model_refs:
        model_refs.remove(base_model)
        ...
        model_refs = list(model_refs)
    base_tensor = tensors.get(base_model, None)
    tensors = [tensors[key] for key in model_refs]
```

This produces:
- `base_tensor` = $\theta^{\text{SFT}} \in \mathbb{R}^{d_1 \times d_2}$ (the
  base model's tensor)
- `tensors` = $[\theta_1^{\text{RL}}, \theta_2^{\text{RL}}, \dots,
  \theta_N^{\text{RL}}]$ (expert tensors, base excluded)

Per-model parameters (`weight`, `density`) are collected into parallel lists:

```python
# mergekit/merge_methods/easy_define.py:162-165
for key in tensor_parameters:
    inner_kwargs[key.name] = [
        self.tensor_parameters[ref][key.name] for ref in model_refs
    ]
```

---

## Step 1: Task Vector Computation

$$
\tau_i = \theta_i^{\text{RL}} - \theta^{\text{SFT}}, \quad i = 1, \dots, N
$$

```python
# longcat_merge/nde.py:37
task_vectors = [t.to(base_tensor.dtype) - base_tensor for t in tensors]
```

Each $\tau_i$ has the same shape as the weight tensor (e.g., $\mathbb{R}^{d_1
\times d_2}$). The `.to(base_tensor.dtype)` ensures all experts match the base
dtype before subtraction.

---

## Step 2: Normalize — Balance Domain Contributions

**Goal**: Scale every task vector to the same L2 norm so no single domain
dominates due to larger parameter deltas.

### 2a. Compute per-expert L2 norms

$$
n_i = \|\tau_i\|_2 = \sqrt{\sum_j (\tau_i)_j^2}
$$

```python
# longcat_merge/nde.py:40
norms = [tv.float().norm() for tv in task_vectors]
```

The `.float()` upcast avoids overflow in float16/bfloat16 during the norm
computation. `torch.Tensor.norm()` with no arguments computes the Frobenius
norm, which for a vector is the L2 norm:
$\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$.

### 2b. Compute the mean norm

$$
\bar{n} = \frac{1}{N} \sum_{i=1}^{N} n_i
$$

```python
# longcat_merge/nde.py:41
mean_norm = sum(norms) / len(norms)
```

### 2c. Rescale each task vector

$$
\hat{\tau}_i = \tau_i \cdot \frac{\bar{n}}{n_i}, \quad \text{if } n_i > \epsilon
$$

where $\epsilon = 10^{-8}$ guards against division by zero (skip normalization
for zero-magnitude task vectors).

```python
# longcat_merge/nde.py:42-44
for i, norm in enumerate(norms):
    if norm > 1e-8:
        task_vectors[i] = task_vectors[i] * (mean_norm / norm)
```

After this step, $\|\hat{\tau}_i\|_2 = \bar{n}$ for all experts (assuming
$n_i > \epsilon$). The direction of each task vector is preserved; only the
magnitude is equalized.

---

## Step 3: Dropout — DARE-style Bernoulli Pruning

**Goal**: Randomly zero out a fraction of parameters to reduce interference
between experts, following DARE (Yu et al., 2024a).

### 3a. Generate Bernoulli mask

For each expert $i$ with keep-probability $p_i$ (the `density` parameter),
sample a binary mask:

$$
m_{ij} \sim \text{Bernoulli}(p_i), \quad j = 1, \dots, d
$$

```python
# mergekit/sparsify.py:131-133
mask = torch.bernoulli(
    torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
)
```

### 3b. Apply mask

$$
\tilde{\tau}_i^{\text{masked}} = \hat{\tau}_i \odot m_i
$$

where $\odot$ denotes element-wise multiplication.

```python
# mergekit/sparsify.py:37
masked = tensor * mask
```

### 3c. L1 rescaling

After dropout, many elements are zero, so $\|\tilde{\tau}_i^{\text{masked}}\|_1
< \|\hat{\tau}_i\|_1$. To compensate, rescale so the L1 norm is preserved:

$$
\tilde{\tau}_i = \tilde{\tau}_i^{\text{masked}} \cdot \frac{\|\hat{\tau}_i\|_1}{\|\tilde{\tau}_i^{\text{masked}}\|_1}
$$

```python
# mergekit/sparsify.py:40-42
elif norm == RescaleNorm.l1:
    before_scale = tensor.abs().sum()       # ||tau_hat_i||_1
    after_scale = masked.abs().sum()        # ||tau_masked_i||_1

# mergekit/sparsify.py:53
return masked * (before_scale / after_scale)
```

The guard at line 51 (`if before_scale < eps or after_scale < eps: return
masked`) handles near-zero tensors to avoid division by zero.

The full dropout call from our code:

```python
# longcat_merge/nde.py:47-51
for i, d in enumerate(density):
    if d < 1.0:
        task_vectors[i] = bernoulli(
            task_vectors[i], density=d, rescale_norm=RescaleNorm.l1
        )
```

**Note on expected value**: In expectation, $\mathbb{E}[m_{ij}] = p_i$, so
$\mathbb{E}[\|\tilde{\tau}_i^{\text{masked}}\|_1] \approx p_i
\|\hat{\tau}_i\|_1$. The L1 rescaling factor is approximately $1/p_i$,
compensating for the dropped elements. However, the actual rescaling uses the
_realized_ norms, not the expected ones, so it adapts to the specific random
draw.

---

## Step 4: Erase — Sign Consensus Mask (SCE-style)

**Goal**: At each parameter position, determine the "majority direction" of
updates across experts. Erase (zero out) any expert whose update disagrees
with the majority. This resolves sign conflicts that cause destructive
interference.

### 4a. Stack task vectors

$$
\Delta \in \mathbb{R}^{N \times d_1 \times d_2}, \quad
\Delta[i, :, :] = \tilde{\tau}_i
$$

```python
# longcat_merge/nde.py:54
stacked = torch.stack(task_vectors, dim=0)
```

### 4b. Compute element-wise signs

$$
s_{ij} = \text{sign}(\Delta[i, j]) = \begin{cases}
+1 & \text{if } \Delta[i,j] > 0 \\
0 & \text{if } \Delta[i,j] = 0 \\
-1 & \text{if } \Delta[i,j] < 0
\end{cases}
$$

```python
# mergekit/merge_methods/generalized_task_arithmetic.py:243
sign = delta.sign().to(mask_dtype)
```

### 4c. Determine majority sign (sum method)

Sum the actual values (not just signs) across experts at each position. The
sign of this sum determines the majority direction, weighted by magnitude:

$$
S_j = \sum_{i=1}^{N} \Delta[i, j]
$$

$$
\sigma_j = \text{sign}(S_j) = \begin{cases}
+1 & \text{if } S_j \geq 0 \\
-1 & \text{if } S_j < 0
\end{cases}
$$

```python
# mergekit/merge_methods/generalized_task_arithmetic.py:245-247
if method == "sum":
    sign_weight = delta.sum(dim=0)                          # S_j
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1  # sigma_j
```

The expression `(x >= 0) * 2 - 1` maps `True→+1` and `False→-1`.

**Important**: The "sum" method means larger-magnitude deltas have more
influence on determining the majority direction. An expert with a large
positive delta can outweigh two experts with small negative deltas.

### 4d. Build the erase mask

$$
E_{ij} = \begin{cases}
1 & \text{if } s_{ij} = \sigma_j \quad \text{(agrees with majority)} \\
0 & \text{if } s_{ij} \neq \sigma_j \quad \text{(minority — erased)}
\end{cases}
$$

```python
# mergekit/merge_methods/generalized_task_arithmetic.py:254
return sign == majority_sign    # boolean tensor, shape (N, d1, d2)
```

Called from our code:

```python
# longcat_merge/nde.py:55
erase_mask = sign_consensus_mask(stacked, method="sum", mask_dtype=mask_dtype)
```

**Special cases**:
- Elements where $\Delta[i,j] = 0$: `sign()` returns 0, which never equals
  $\pm 1$, so these are always erased (masked out). This is correct — a zero
  delta contributes nothing.
- Elements where all experts agree: All get $E_{ij} = 1$ — no erasure.

---

## Step 5: Weighted Combination with Per-Element Normalization

### 5a. Shape the weight vector

The per-model weights $w_i$ are broadcast to match the tensor dimensions:

$$
W \in \mathbb{R}^{N \times 1 \times 1}, \quad W[i] = w_i
$$

```python
# longcat_merge/nde.py:58-60
weights = torch.tensor(weight, dtype=stacked.dtype, device=stacked.device)
while weights.dim() < stacked.dim():
    weights = weights.unsqueeze(-1)
```

### 5b. Apply weights and erase mask

$$
\hat{W}_{ij} = w_i \cdot E_{ij}
$$

```python
# longcat_merge/nde.py:62
erased_weights = weights * erase_mask
```

$\hat{W}_{ij}$ is zero wherever expert $i$ was erased at position $j$, and
$w_i$ otherwise.

### 5c. Weighted sum of task vectors

$$
\Delta^{\text{merged}}_j = \sum_{i=1}^{N} \hat{W}_{ij} \cdot \tilde{\tau}_{ij}
= \sum_{i=1}^{N} w_i \cdot E_{ij} \cdot \tilde{\tau}_{ij}
$$

```python
# longcat_merge/nde.py:63
merged_delta = (stacked * erased_weights).sum(dim=0)
```

### 5d. Per-element normalization

Divide by the sum of active weights at each position:

$$
D_j = \max\!\left(\sum_{i=1}^{N} \hat{W}_{ij},\; 10^{-6}\right)
$$

$$
\bar{\Delta}_j = \frac{\Delta^{\text{merged}}_j}{D_j}
$$

```python
# longcat_merge/nde.py:64-65
divisor = erased_weights.sum(dim=0).clamp(min=1e-6)
merged_delta = merged_delta / divisor
```

The `clamp(min=1e-6)` prevents division by zero at positions where all experts
were erased (all had minority-direction updates). At such positions, the
numerator $\Delta^{\text{merged}}_j$ is also zero (all contributions masked),
so $\bar{\Delta}_j \approx 0$.

**Why per-element normalization**: Without this division, positions where more
experts survive erasure would have proportionally larger deltas than positions
where fewer survive. Dividing by the active weight sum normalizes the
contribution regardless of how many experts agree.

---

## Step 6: Reconstruct Merged Parameters

$$
\theta^{\text{merged}} = \theta^{\text{SFT}} + \bar{\Delta}
$$

```python
# longcat_merge/nde.py:67
return base_tensor + merged_delta
```

---

## Full Pipeline Summary

Combining all steps, the final merged parameter at element $j$ is:

$$
\theta^{\text{merged}}_j = \theta^{\text{SFT}}_j + \frac{
  \displaystyle\sum_{i=1}^{N} w_i \cdot E_{ij} \cdot \tilde{\tau}_{ij}
}{
  \displaystyle\sum_{i=1}^{N} w_i \cdot E_{ij}
}
$$

where each $\tilde{\tau}_{ij}$ is the result of applying normalization then
dropout to the raw task vector:

$$
\tilde{\tau}_i = \text{L1Rescale}\!\Big(
  \underbrace{\tau_i \cdot \frac{\bar{n}}{n_i}}_{\text{normalized}}
  \;\odot\; m_i
\Big)
$$

and each $E_{ij}$ is the sign consensus erase mask:

$$
E_{ij} = \mathbb{1}\!\big[\text{sign}(\tilde{\tau}_{ij}) = \text{sign}\!\big(\textstyle\sum_k \tilde{\tau}_{kj}\big)\big]
$$

---

## Degenerate Case: $N = 1$ (Single Expert)

When there is only one RL checkpoint and one base model:

1. **Normalize**: $\bar{n} = n_1$, so $\hat{\tau}_1 = \tau_1 \cdot
   \frac{n_1}{n_1} = \tau_1$. **No-op.**

2. **Dropout**: Bernoulli mask is applied, then L1-rescaled. **This still
   has effect** — it randomly zeros elements and rescales.

3. **Erase**: With $N=1$, every element's sign trivially agrees with the
   "majority" (itself). Formally: $\sigma_j = \text{sign}(\tilde{\tau}_{1j})$
   and $s_{1j} = \text{sign}(\tilde{\tau}_{1j})$, so $E_{1j} = 1$ everywhere
   (except where $\tilde{\tau}_{1j} = 0$, which are already zero).
   **No-op on non-zero elements.**

4. **Weighted combination**: $\bar{\Delta}_j = \frac{w_1 \cdot 1 \cdot
   \tilde{\tau}_{1j}}{w_1 \cdot 1} = \tilde{\tau}_{1j}$. The weight cancels.

**Net effect with $N=1$**: The merge reduces to
$\theta^{\text{merged}} = \theta^{\text{SFT}} + \text{L1Rescale}(\tau_1 \odot m_1)$
— i.e., DARE-style dropout applied to the single task vector. The Normalize
and Erase steps become identity operations. This is equivalent to
`dare_linear` with `rescale=True, normalize=False`.
