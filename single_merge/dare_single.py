"""
DARE-linear single-model merge. Edit the vars below and run:
  python single_merge/dare_single.py

Uses mergekit slices to exclude MTP layer from merge.
RL checkpoint does NOT need MTP inserted — slices only reference RL for layers 0..NUM_HIDDEN-1.
"""

from mergekit.config import (
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.merge import MergeOptions, run_merge

# ---- edit these ----
BASE_MODEL = "/root/myCodeLab/host/downloads/models/40Bv6/dpo-0210-0208-v2-dpoaddid-965/965"
RL_MODEL = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c351_facpo_nemogym_math_d0.5-tp1.5-tn2.0-ent0-bdm1-ppoch2-1cb2094f_20260213_184907/global_step_80/actor/huggingface"
OUT_PATH = "./merged_output"
DTYPE = "bfloat16"
CUDA = False

NUM_HIDDEN = 40   # base transformer layers: 0..39
EXCLUDE_LAYERS = [40]  # MTP layer index

COMBO = "recommended"  # <-- switch combo here

COMBOS = {
    "recommended":  {"weight": 0.7, "density": 0.5},
    "conservative": {"weight": 0.5, "density": 0.3},
    "aggressive":   {"weight": 1.0, "density": 0.7},
}
# ---------------------

params = COMBOS[COMBO]
w, d = params["weight"], params["density"]

# Build layer ranges: merge everything except excluded layers
# For layers 0..39: both base + RL (dare_linear merge)
# For layer 40 (MTP): base only (passes through unchanged)
slices = []

# merged range: [0, NUM_HIDDEN)
slices.append(
    OutputSliceDefinition(
        sources=[
            InputSliceDefinition(model=BASE_MODEL, layer_range=[0, NUM_HIDDEN]),
            InputSliceDefinition(model=RL_MODEL, layer_range=[0, NUM_HIDDEN],
                                 parameters={"weight": w, "density": d}),
        ],
    )
)

# excluded layers: base only, dare_linear sees no task vectors → returns base
for layer_idx in EXCLUDE_LAYERS:
    slices.append(
        OutputSliceDefinition(
            sources=[
                InputSliceDefinition(model=BASE_MODEL, layer_range=[layer_idx, layer_idx + 1]),
            ],
        )
    )

config = MergeConfiguration(
    merge_method="dare_linear",
    base_model=BASE_MODEL,
    slices=slices,
    dtype=DTYPE,
)

print(f"[{COMBO}] weight={w}, density={d}")
print(f"  base:    {BASE_MODEL}")
print(f"  rl:      {RL_MODEL}")
print(f"  merge:   layers 0-{NUM_HIDDEN - 1}")
print(f"  exclude: layers {EXCLUDE_LAYERS} (copy from base)")
print(f"  out:     {OUT_PATH}")

run_merge(config, OUT_PATH, options=MergeOptions(cuda=CUDA))
print("Done.")
