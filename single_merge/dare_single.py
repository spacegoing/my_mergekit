"""
DARE-linear single-model merge. Edit the vars below and run:
  python single_merge/dare_single.py

Supports:
  - Layer-selective merging (MERGE_RANGE)
  - Component-selective merging (SKIP_GATE)
  - MTP layer exclusion
"""

from mergekit.config import (
    ConditionalParameter,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.merge import MergeOptions, run_merge

# ---- edit these ----
BASE_MODEL = "/root/myCodeLab/host/downloads/models/40Bv6/dpo-0210-0208-v2-dpoaddid-965/965"
RL_MODEL = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c415_facpo_nemogym_math_d1.0-tp1.5-tn2.0-ent0.001-bdm1-ppoch2-575c58dc_20260215_033228/global_step_30/actor/huggingface"

RL_MODEL = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c351_facpo_nemogym_math_d0.5-tp1.5-tn2.0-ent0-bdm1-ppoch2-1cb2094f_20260213_184907/global_step_80/actor/huggingface"
RL_MODEL = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c411_facpo_nemogym_math_d1.0-tp1.5-tn2.0-ent0-bdm1-ppoch2-575c58dc_20260214_211417/global_step_60/actor/huggingface"
OUT_PATH = "./c411_kgs6v3"
COMBO = "ta_0.3"  # <-- switch combo here
# Component-selective: skip moe_gate weights (keep routing from base).
# analyze_delta.py showed moe_gate has 10-100x larger relative change than other components.
SKIP_GATE = True  # <-- set False to merge everything including gates

DTYPE = "float32"       # compute in float32 for precision (bf16 loses bits in τ = θ_RL - θ_base)
OUT_DTYPE = "bfloat16"  # save output in bfloat16 to keep model size normal
CUDA = True

NUM_HIDDEN = 40   # base transformer layers: 0..39
MTP_LAYER = 40    # MTP layer index (always from base)

# Layer-selective: which layers to merge. Layers outside stay pure base.
MERGE_RANGE = [0, 40]   # <-- [start, end), e.g. [20,40] to skip early layers

COMBOS = {
    # task arithmetic (density=1.0, no dropout) — best for single expert on large models
    "ta_0.05":         {"weight": 0.05, "density": 1.0},
    "ta_0.1":         {"weight": 0.1, "density": 1.0},
    "ta_0.1_8":         {"weight": 0.1, "density": 0.8},
    "ta_0.2":         {"weight": 0.2, "density": 1.0},
    "ta_0.3":         {"weight": 0.3, "density": 1.0},
    "ta_0.5":         {"weight": 0.5, "density": 1.0},
    "ta_0.7":         {"weight": 0.7, "density": 1.0},
    "ta_1.0":         {"weight": 1.0, "density": 1.0},
    # dare (with dropout) — for reference, not recommended for 40B MoE
    "dare_0.7_d0.5":  {"weight": 0.7, "density": 0.5},
}
# ---------------------

params = COMBOS[COMBO]
w, d = params["weight"], params["density"]
merge_start, merge_end = MERGE_RANGE

# Build per-model parameters with optional gate skipping
if SKIP_GATE:
    # weight=0 for gate tensors → keeps base routing; weight=w for everything else
    rl_params = {
        "weight": [
            ConditionalParameter(value=0.0, filter=".mlp.gate."),
            ConditionalParameter(value=w),
        ],
        "density": d,
    }
else:
    rl_params = {"weight": w, "density": d}

# Build slices: base-only → merged → base-only → MTP
slices = []

# 1. Layers before merge range: base only
if merge_start > 0:
    slices.append(
        OutputSliceDefinition(sources=[
            InputSliceDefinition(model=BASE_MODEL, layer_range=[0, merge_start]),
        ])
    )

# 2. Merged range: base + RL
slices.append(
    OutputSliceDefinition(sources=[
        InputSliceDefinition(model=BASE_MODEL, layer_range=[merge_start, merge_end]),
        InputSliceDefinition(model=RL_MODEL, layer_range=[merge_start, merge_end],
                             parameters=rl_params),
    ])
)

# 3. Layers after merge range but before MTP: base only
if merge_end < NUM_HIDDEN:
    slices.append(
        OutputSliceDefinition(sources=[
            InputSliceDefinition(model=BASE_MODEL, layer_range=[merge_end, NUM_HIDDEN]),
        ])
    )

# 4. MTP layer: base only
slices.append(
    OutputSliceDefinition(sources=[
        InputSliceDefinition(model=BASE_MODEL, layer_range=[MTP_LAYER, MTP_LAYER + 1]),
    ])
)

config = MergeConfiguration(
    merge_method="dare_linear",
    base_model=BASE_MODEL,
    slices=slices,
    dtype=DTYPE,
    out_dtype=OUT_DTYPE,
)

gate_str = "SKIP moe_gate (weight=0)" if SKIP_GATE else "merge all components"
layer_str = f"layers {merge_start}-{merge_end - 1}"
if merge_start > 0:
    layer_str += f"  (skip 0-{merge_start - 1} from base)"

print(f"[{COMBO}] weight={w}, density={d}")
print(f"  base:    {BASE_MODEL}")
print(f"  rl:      {RL_MODEL}")
print(f"  merge:   {layer_str}")
print(f"  gate:    {gate_str}")
print(f"  base:    layers outside [{merge_start}, {merge_end}) + MTP layer {MTP_LAYER}")
print(f"  out:     {OUT_PATH}")

run_merge(config, OUT_PATH, options=MergeOptions(cuda=CUDA))
print("Done.")
