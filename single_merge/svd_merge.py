#!/usr/bin/env python3
"""
SVD-based component-aware model merge.

Different components get different treatment based on delta analysis:
  - moe_gate:       skip (weight=0), large delta but not beneficial in single-expert merge
  - expert_mlp:     SVD low-rank denoising + higher weight (0.02% relative Δ → mostly noise)
  - attention:      regular task arithmetic
  - shared_expert:  regular task arithmetic
  - norm:           skip (delta ≈ 0)
  - other:          regular task arithmetic (embeddings, lm_head)

Usage:
  # Direct (edit defaults below):
  python single_merge/svd_merge.py

  # With CLI args (for batch runner):
  python single_merge/svd_merge.py --base_dir /path/to/base --rl_dir /path/to/rl \
      --out_dir ./output --combo b --device cuda
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ===================== DEFAULTS (overridden by CLI) =====================
BASE_DIR = "/root/myCodeLab/host/downloads/models/40Bv6/dpo-0210-0208-v2-dpoaddid-965/965"
RL_DIR = "/root/myCodeLab/host/verl/ckpts/single_domain/sd_c351_facpo_nemogym_math_d0.5-tp1.5-tn2.0-ent0-bdm1-ppoch2-1cb2094f_20260213_184907/global_step_80/actor/huggingface"
OUT_DIR = "./merged_output"
COMBO = "b"

# Combos: experiment-informed hyperparameter presets
# Rationale:
#   - Uniform TA w/o gate: v3(0.3) too conservative, v5(0.5) best balance, v7(0.7) best math
#   - SVD denoises expert MLP → can push expert weight higher than uniform
#   - Gate skip proven beneficial in single-expert merge
COMBOS = {
    # Conservative: close to TA v3, minimal risk
    "c": {"gate": 0.0, "expert": 0.5, "attn": 0.3, "shared": 0.3, "other": 0.2, "rank": 32},
    # Balanced: close to TA v5, SVD lets expert go higher
    "b": {"gate": 0.0, "expert": 0.7, "attn": 0.5, "shared": 0.5, "other": 0.3, "rank": 32},
    # Aggressive: maximize RL gains, tighter SVD denoising
    "a": {"gate": 0.0, "expert": 1.0, "attn": 0.7, "shared": 0.7, "other": 0.5, "rank": 16},
}

MTP_LAYER = 40
OUT_DTYPE = torch.bfloat16
# ========================================================================


def load_index(model_dir):
    path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(path) as f:
        return json.load(f)


def classify(key):
    """Classify tensor into component type."""
    if ".mlp.gate." in key:
        return "gate"
    if ".mlp.experts." in key:
        return "expert_mlp"
    if ".mlp.shared_experts." in key:
        return "shared_expert"
    if ".mlp." in key:
        return "dense_mlp"
    if ".self_attn." in key:
        return "attention"
    if "layernorm" in key or "norm" in key:
        return "norm"
    return "other"


def parse_layer(key):
    m = re.match(r"model\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def svd_low_rank(delta, rank, device="cpu"):
    """Low-rank approximation via truncated SVD. Returns (denoised_delta, energy)."""
    if delta.dim() != 2 or rank <= 0:
        return delta, 1.0
    r = min(rank, min(delta.shape))
    # Move to GPU for faster SVD if available
    delta_dev = delta.to(device)
    U, S, Vh = torch.linalg.svd(delta_dev, full_matrices=False)
    # Energy captured: sum(S[:r]^2) / sum(S^2)
    total_energy = S.pow(2).sum()
    energy = S[:r].pow(2).sum() / total_energy if total_energy > 0 else 1.0
    result = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    return result.cpu() if device != "cpu" else result, energy.item()


def parse_args():
    parser = argparse.ArgumentParser(description="SVD-based component-aware model merge")
    parser.add_argument("--base_dir", default=BASE_DIR, help="Base model directory")
    parser.add_argument("--rl_dir", default=RL_DIR, help="RL checkpoint directory")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output directory")
    parser.add_argument("--combo", default=COMBO, choices=list(COMBOS.keys()),
                        help="Hyperparameter combo: c(onservative), b(alanced), a(ggressive)")
    parser.add_argument("--device", default="cpu", help="Device for SVD computation (cpu, cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    rl_dir = args.rl_dir
    out_dir = args.out_dir
    # Auto-append combo name if not already present
    if not out_dir.endswith(f"_svd{args.combo}") and not out_dir.endswith(f"svd{args.combo}"):
        out_dir = f"{out_dir}_svd{args.combo}"
    device = args.device
    combo = COMBOS[args.combo]

    gate_w = combo["gate"]
    expert_w = combo["expert"]
    attn_w = combo["attn"]
    shared_w = combo["shared"]
    other_w = combo["other"]
    svd_rank = combo["rank"]

    def get_weight(comp):
        return {
            "gate": gate_w, "expert_mlp": expert_w, "attention": attn_w,
            "shared_expert": shared_w, "dense_mlp": other_w, "norm": 0.0,
            "other": other_w,
        }[comp]

    print(f"SVD-based component-aware merge  [combo={args.combo}]")
    print(f"  base:          {base_dir}")
    print(f"  rl:            {rl_dir}")
    print(f"  out:           {out_dir}")
    print(f"  device:        {device}")
    print(f"  SVD rank:      {svd_rank}")
    print(f"  gate weight:   {gate_w}  (skip)")
    print(f"  expert weight: {expert_w}  (SVD rank={svd_rank})")
    print(f"  attn weight:   {attn_w}")
    print(f"  shared weight: {shared_w}")
    print(f"  other weight:  {other_w}")
    print()

    # Load indices
    base_index = load_index(base_dir)
    rl_index = load_index(rl_dir)
    base_map = base_index["weight_map"]
    rl_map = rl_index["weight_map"]

    all_keys = sorted(base_map.keys())
    print(f"Total keys: {len(all_keys)}")

    # Group keys by BASE shard (output uses same shard structure)
    shard_keys = defaultdict(list)
    for key in all_keys:
        shard_keys[base_map[key]].append(key)

    os.makedirs(out_dir, exist_ok=True)

    # Lazily opened RL shard handles (RL keys may be in different shards)
    rl_handles = {}

    # Stats
    stats = defaultdict(lambda: {"count": 0, "svd_energy": []})
    new_weight_map = {}

    shard_names = sorted(shard_keys.keys())
    for si, shard_name in enumerate(shard_names):
        keys = shard_keys[shard_name]
        print(f"[{si + 1}/{len(shard_names)}] {shard_name} ({len(keys)} tensors)")

        base_f = safe_open(
            os.path.join(base_dir, shard_name), framework="pt", device="cpu"
        )

        merged = {}
        for key in keys:
            base_t = base_f.get_tensor(key).float()
            layer = parse_layer(key)
            comp = classify(key)
            w = get_weight(comp)

            # MTP layer: always keep base
            if layer is not None and layer >= MTP_LAYER:
                merged[key] = base_t.to(OUT_DTYPE)
                new_weight_map[key] = shard_name
                stats["mtp"]["count"] += 1
                continue

            # Skip if weight=0 or key not in RL
            if w == 0.0 or key not in rl_map:
                merged[key] = base_t.to(OUT_DTYPE)
                new_weight_map[key] = shard_name
                stats[comp]["count"] += 1
                continue

            # Load RL tensor (from correct RL shard)
            rl_shard = rl_map[key]
            if rl_shard not in rl_handles:
                rl_handles[rl_shard] = safe_open(
                    os.path.join(rl_dir, rl_shard), framework="pt", device="cpu"
                )
            rl_t = rl_handles[rl_shard].get_tensor(key).float()
            delta = rl_t - base_t

            # Apply SVD for expert MLP
            if comp == "expert_mlp" and svd_rank > 0 and delta.dim() == 2:
                delta, energy = svd_low_rank(delta, svd_rank, device=device)
                stats[comp]["svd_energy"].append(energy)
            else:
                stats[comp]["count"] += 1

            result = base_t + w * delta
            merged[key] = result.to(OUT_DTYPE)
            new_weight_map[key] = shard_name

        # Save shard
        save_file(merged, os.path.join(out_dir, shard_name))
        del merged
        print(f"  saved.")

    # Write index
    out_index = {
        "metadata": base_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(out_index, f, indent=2)

    # Copy config + tokenizer from base
    copy_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
    ]
    for fname in copy_files:
        src = os.path.join(base_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            print(f"  copied {fname}")

    # Print stats
    print("\n" + "=" * 60)
    print(f"Component stats:  [combo={args.combo}]")
    for comp in ["gate", "expert_mlp", "attention", "shared_expert",
                 "dense_mlp", "norm", "other", "mtp"]:
        s = stats[comp]
        cnt = s["count"]
        energies = s["svd_energy"]
        if cnt == 0 and not energies:
            continue
        total = cnt + len(energies)
        line = f"  {comp:<15s}  {total:>6d} tensors  weight={get_weight(comp) if comp != 'mtp' else 0.0}"
        if energies:
            avg_e = sum(energies) / len(energies)
            min_e = min(energies)
            line += f"  SVD: rank={svd_rank}, avg_energy={avg_e:.4f}, min={min_e:.4f}"
        print(line)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
