"""NDE (Normalize-Dropout-Erase) merge method.

Three-pronged strategy for merging domain-expert RL models:
  1. Normalize task vector magnitudes to balance domain contributions
  2. Dropout (DARE-style) to prune redundant delta parameters
  3. Erase (SCE-style) minority-direction parameter updates
"""

from typing import List

import torch

from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)
from mergekit.sparsify import RescaleNorm, bernoulli


@merge_method(
    name="nde",
    pretty_name="NDE (Normalize-Dropout-Erase)",
)
def nde_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    weight: List[float],
    density: List[float] = 1.0,
    int8_mask: bool = False,
) -> torch.Tensor:
    if not tensors:
        return base_tensor

    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype

    # Step 1: compute task vectors (tau_i = theta_RL_i - theta_SFT)
    task_vectors = [t.to(base_tensor.dtype) - base_tensor for t in tensors]

    # Step 2: normalize — scale each task vector to the mean L2 norm
    norms = [tv.float().norm() for tv in task_vectors]
    mean_norm = sum(norms) / len(norms)
    for i, norm in enumerate(norms):
        if norm > 1e-8:
            task_vectors[i] = task_vectors[i] * (mean_norm / norm)

    # Step 3: dropout — DARE-style Bernoulli dropout with L1 rescaling
    for i, d in enumerate(density):
        if d < 1.0:
            task_vectors[i] = bernoulli(
                task_vectors[i], density=d, rescale_norm=RescaleNorm.l1
            )

    # Step 4: erase — SCE-style sign consensus mask
    stacked = torch.stack(task_vectors, dim=0)
    erase_mask = sign_consensus_mask(stacked, method="sum", mask_dtype=mask_dtype)

    # Step 5: weighted combination with per-element normalization
    weights = torch.tensor(weight, dtype=stacked.dtype, device=stacked.device)
    while weights.dim() < stacked.dim():
        weights = weights.unsqueeze(-1)

    erased_weights = weights * erase_mask
    merged_delta = (stacked * erased_weights).sum(dim=0)
    divisor = erased_weights.sum(dim=0).clamp(min=1e-6)
    merged_delta = merged_delta / divisor

    return base_tensor + merged_delta
