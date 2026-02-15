"""
Analyze per-layer, per-component delta magnitudes between base and RL checkpoints.
Verifies which layers/components changed the most during RL training.

Usage:
  python single_merge/analyze_delta.py

Outputs a table showing L2 norm of (θ_RL - θ_base) grouped by layer and component type.
"""

import json
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open

# ---- edit these (same as dare_single.py) ----
BASE_DIR = "/root/myCodeLab/host/downloads/models/40Bv6/dpo-0210-0208-v2-dpoaddid-965/965"
RL_DIR = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c351_facpo_nemogym_math_d0.5-tp1.5-tn2.0-ent0-bdm1-ppoch2-1cb2094f_20260213_184907/global_step_80/actor/huggingface"
# ----------------------------------------------


def load_index(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def classify_component(suffix):
    """Classify a weight suffix into a component type."""
    if ".mlp.gate." in suffix:
        return "moe_gate"
    if ".mlp.experts." in suffix:
        return "expert_mlp"
    if ".mlp.shared_experts." in suffix:
        return "shared_expert"
    if ".mlp." in suffix:  # dense MLP (layer 0)
        return "dense_mlp"
    if ".self_attn." in suffix:
        return "attention"
    if "layernorm" in suffix or "norm" in suffix:
        return "norm"
    return "other"


def parse_layer(key):
    """Extract layer index from key. Returns (layer_idx, suffix) or (None, key)."""
    m = re.match(r"model\.layers\.(\d+)\.(.*)", key)
    if m:
        return int(m.group(1)), m.group(2)
    return None, key


def main():
    print("Loading indices...")
    base_map = load_index(BASE_DIR)
    rl_map = load_index(RL_DIR)

    # Only compare keys present in both
    common_keys = sorted(set(base_map.keys()) & set(rl_map.keys()))
    print(f"Common keys: {len(common_keys)}")

    # Collect per-(layer, component) stats
    # Store: sum of squared deltas (for L2 norm), sum of squared base (for relative norm)
    delta_sq = defaultdict(float)     # (layer, component) -> sum ||delta||^2
    base_sq = defaultdict(float)      # (layer, component) -> sum ||base||^2
    param_count = defaultdict(int)    # (layer, component) -> num params

    # Cache open file handles
    base_handles = {}
    rl_handles = {}

    print("Computing deltas (this reads all tensors — may take a while)...")
    for i, key in enumerate(common_keys):
        if i % 500 == 0:
            print(f"  [{i}/{len(common_keys)}] {key[:60]}...")

        layer_idx, suffix = parse_layer(key)
        component = classify_component(key)
        group = (layer_idx, component)  # layer_idx=None for non-layer weights

        # Open shard files lazily
        base_shard = base_map[key]
        if base_shard not in base_handles:
            base_handles[base_shard] = safe_open(
                os.path.join(BASE_DIR, base_shard), framework="pt", device="cpu"
            )
        rl_shard = rl_map[key]
        if rl_shard not in rl_handles:
            rl_handles[rl_shard] = safe_open(
                os.path.join(RL_DIR, rl_shard), framework="pt", device="cpu"
            )

        base_t = base_handles[base_shard].get_tensor(key).float()
        rl_t = rl_handles[rl_shard].get_tensor(key).float()
        delta = rl_t - base_t

        delta_sq[group] += delta.pow(2).sum().item()
        base_sq[group] += base_t.pow(2).sum().item()
        param_count[group] += delta.numel()

    # Aggregate by layer
    print("\n" + "=" * 90)
    print(f"{'Layer':>5}  {'Component':<15}  {'||delta||_2':>12}  {'||base||_2':>12}  "
          f"{'relative%':>10}  {'#params':>10}")
    print("-" * 90)

    # Sort by layer, then component
    all_groups = sorted(delta_sq.keys(), key=lambda x: (x[0] if x[0] is not None else -1, x[1]))

    for group in all_groups:
        layer_idx, component = group
        d_norm = delta_sq[group] ** 0.5
        b_norm = base_sq[group] ** 0.5
        rel = (d_norm / b_norm * 100) if b_norm > 0 else 0
        layer_str = str(layer_idx) if layer_idx is not None else "—"
        print(f"{layer_str:>5}  {component:<15}  {d_norm:>12.4f}  {b_norm:>12.4f}  "
              f"{rel:>9.4f}%  {param_count[group]:>10}")

    # Summary: per-layer totals
    print("\n" + "=" * 90)
    print("Per-layer total delta (all components combined):")
    print(f"{'Layer':>5}  {'||delta||_2':>12}  {'||base||_2':>12}  {'relative%':>10}")
    print("-" * 50)

    layer_delta = defaultdict(float)
    layer_base = defaultdict(float)
    for (layer_idx, comp), val in delta_sq.items():
        if layer_idx is not None:
            layer_delta[layer_idx] += val
            layer_base[layer_idx] += base_sq[(layer_idx, comp)]

    for layer_idx in sorted(layer_delta.keys()):
        d_norm = layer_delta[layer_idx] ** 0.5
        b_norm = layer_base[layer_idx] ** 0.5
        rel = (d_norm / b_norm * 100) if b_norm > 0 else 0
        print(f"{layer_idx:>5}  {d_norm:>12.4f}  {b_norm:>12.4f}  {rel:>9.4f}%")

    # Top-10 components by relative change
    print("\n" + "=" * 90)
    print("Top-20 (layer, component) by relative delta:")
    print(f"{'Layer':>5}  {'Component':<15}  {'||delta||_2':>12}  {'relative%':>10}")
    print("-" * 55)

    ranked = sorted(
        delta_sq.keys(),
        key=lambda g: (delta_sq[g] ** 0.5 / max(base_sq[g] ** 0.5, 1e-12)),
        reverse=True,
    )
    for group in ranked[:20]:
        layer_idx, component = group
        d_norm = delta_sq[group] ** 0.5
        b_norm = base_sq[group] ** 0.5
        rel = (d_norm / b_norm * 100) if b_norm > 0 else 0
        layer_str = str(layer_idx) if layer_idx is not None else "—"
        print(f"{layer_str:>5}  {component:<15}  {d_norm:>12.4f}  {rel:>9.4f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
