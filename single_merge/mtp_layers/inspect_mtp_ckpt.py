#!/usr/bin/env python3
"""Inspect a DeepSeek-V3 checkpoint to understand MTP layer structure.

Reads model.safetensors.index.json and optionally probes tensor shapes/dtypes
to confirm MTP layer presence and weight structure.

Usage:
    python inspect_mtp_ckpt.py /path/to/model [--probe]

    --probe: actually open safetensors files to read tensor shapes/dtypes
             (slower but gives full picture)
"""

import argparse
import json
import os
import re
from collections import defaultdict


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)


def load_index(model_dir: str) -> dict | None:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        return json.load(f)


def classify_key(key: str, num_hidden_layers: int) -> str:
    """Classify a weight key into a category."""
    mtp_idx = num_hidden_layers  # MTP layer sits right after base layers

    if key == "model.embed_tokens.weight":
        return "embedding"
    if key == "model.norm.weight":
        return "final_norm"
    if key == "lm_head.weight":
        return "lm_head"

    m = re.match(r"model\.layers\.(\d+)\.", key)
    if m:
        layer_idx = int(m.group(1))
        if layer_idx < num_hidden_layers:
            return f"base_layer"
        elif layer_idx == mtp_idx:
            return "mtp_layer"
        else:
            return f"unknown_layer_{layer_idx}"

    return "other"


def extract_layer_info(key: str) -> tuple[int | None, str]:
    """Extract (layer_index, suffix) from a weight key."""
    m = re.match(r"model\.layers\.(\d+)\.(.*)", key)
    if m:
        return int(m.group(1)), m.group(2)
    return None, key


def main():
    parser = argparse.ArgumentParser(description="Inspect MTP checkpoint structure")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--probe", action="store_true",
                        help="Open safetensors to read tensor shapes/dtypes")
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Model directory: {model_dir}\n")

    # ── Config ────────────────────────────────────────────────────────────
    config = load_config(model_dir)
    if config:
        num_hidden = config.get("num_hidden_layers", "?")
        num_mtp = config.get("num_nextn_predict_layers", "NOT SET")
        n_experts = config.get("n_routed_experts", "?")
        model_type = config.get("model_type", "?")
        print(f"Config:")
        print(f"  model_type:               {model_type}")
        print(f"  num_hidden_layers:        {num_hidden}")
        print(f"  num_nextn_predict_layers: {num_mtp}")
        print(f"  n_routed_experts:         {n_experts}")
        print(f"  hidden_size:              {config.get('hidden_size', '?')}")
        print(f"  tie_word_embeddings:      {config.get('tie_word_embeddings', '?')}")
        num_hidden_layers = num_hidden if isinstance(num_hidden, int) else 40
    else:
        print("WARNING: no config.json found")
        num_hidden_layers = 40

    mtp_layer_idx = num_hidden_layers
    print(f"\n  => Expected MTP layer index: {mtp_layer_idx}")
    print(f"  => Base layers: 0..{num_hidden_layers - 1}")

    # ── Index ─────────────────────────────────────────────────────────────
    index = load_index(model_dir)
    if index is None:
        print("\nERROR: no model.safetensors.index.json found")
        return

    weight_map = index["weight_map"]
    metadata = index.get("metadata", {})
    total_size = metadata.get("total_size", "?")

    print(f"\nIndex:")
    print(f"  total keys:  {len(weight_map)}")
    print(f"  total_size:  {total_size}")

    # ── Shard files ───────────────────────────────────────────────────────
    shard_files = sorted(set(weight_map.values()))
    print(f"\nShard files ({len(shard_files)}):")
    shard_key_counts = defaultdict(int)
    for v in weight_map.values():
        shard_key_counts[v] += 1
    for sf in shard_files:
        fpath = os.path.join(model_dir, sf)
        fsize = os.path.getsize(fpath) if os.path.exists(fpath) else -1
        print(f"  {sf:50s}  keys={shard_key_counts[sf]:4d}  size={fsize / 1e9:.2f} GB")

    # ── Category breakdown ────────────────────────────────────────────────
    categories = defaultdict(list)
    for key in sorted(weight_map.keys()):
        cat = classify_key(key, num_hidden_layers)
        categories[cat].append(key)

    print(f"\nWeight categories:")
    for cat in ["embedding", "final_norm", "lm_head", "base_layer", "mtp_layer", "other"]:
        if cat in categories:
            print(f"  {cat:20s}: {len(categories[cat]):5d} keys")
    # Print unknowns
    for cat in sorted(categories):
        if cat.startswith("unknown_"):
            print(f"  {cat:20s}: {len(categories[cat]):5d} keys")

    # ── Layer index range ─────────────────────────────────────────────────
    layer_indices = set()
    for key in weight_map:
        idx, _ = extract_layer_info(key)
        if idx is not None:
            layer_indices.add(idx)

    if layer_indices:
        print(f"\nLayer indices present: {min(layer_indices)}..{max(layer_indices)}")
        print(f"  Total unique layer indices: {len(layer_indices)}")
        if mtp_layer_idx in layer_indices:
            print(f"  MTP layer {mtp_layer_idx}: PRESENT")
        else:
            print(f"  MTP layer {mtp_layer_idx}: MISSING")

    # ── MTP layer detail ──────────────────────────────────────────────────
    mtp_keys = categories.get("mtp_layer", [])
    if mtp_keys:
        print(f"\nMTP layer keys ({len(mtp_keys)}):")

        # Group by shard
        shard_groups = defaultdict(list)
        for k in mtp_keys:
            shard_groups[weight_map[k]].append(k)

        for shard, keys in sorted(shard_groups.items()):
            print(f"\n  [{shard}] ({len(keys)} keys)")
            for k in sorted(keys):
                print(f"    {k}")
    else:
        print(f"\nNO MTP layer keys found (layer {mtp_layer_idx})")

    # ── Base layer sample (layer 0) ───────────────────────────────────────
    layer0_keys = [k for k in weight_map if k.startswith("model.layers.0.")]
    if layer0_keys:
        print(f"\nBase layer 0 structure ({len(layer0_keys)} keys):")
        for k in sorted(layer0_keys):
            suffix = k.replace("model.layers.0.", "")
            print(f"  .{suffix}")

    # Also show last base layer for comparison
    last_base = num_hidden_layers - 1
    last_keys = [k for k in weight_map if k.startswith(f"model.layers.{last_base}.")]
    if last_keys and last_base != 0:
        print(f"\nLast base layer {last_base} structure ({len(last_keys)} keys):")
        for k in sorted(last_keys):
            suffix = k.replace(f"model.layers.{last_base}.", "")
            print(f"  .{suffix}")

    # ── Shared weight detection ───────────────────────────────────────────
    if mtp_keys:
        mtp_suffixes = set()
        for k in mtp_keys:
            _, suffix = extract_layer_info(k)
            mtp_suffixes.add(suffix)

        # Check for shared-looking keys
        shared_candidates = [
            (f"model.layers.{mtp_layer_idx}.embed_tokens.weight", "model.embed_tokens.weight"),
            (f"model.layers.{mtp_layer_idx}.shared_head.head.weight", "lm_head.weight"),
        ]
        print(f"\nShared weight check:")
        for mtp_k, base_k in shared_candidates:
            mtp_exists = mtp_k in weight_map
            base_exists = base_k in weight_map
            print(f"  {mtp_k:60s} {'EXISTS' if mtp_exists else 'MISSING':8s}  (base: {base_k} {'EXISTS' if base_exists else 'MISSING'})")

    # ── Optional: probe tensor shapes ─────────────────────────────────────
    if args.probe and mtp_keys:
        try:
            from safetensors import safe_open
        except ImportError:
            print("\nERROR: safetensors not installed, cannot probe")
            return

        print(f"\n{'='*70}")
        print("Probing MTP tensor shapes and dtypes...")
        print(f"{'='*70}")

        shard_groups = defaultdict(list)
        for k in mtp_keys:
            shard_groups[weight_map[k]].append(k)

        for shard, keys in sorted(shard_groups.items()):
            fpath = os.path.join(model_dir, shard)
            if not os.path.exists(fpath):
                print(f"\n  [{shard}] FILE NOT FOUND")
                continue
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for k in sorted(keys):
                    tensor = f.get_tensor(k)
                    print(f"  {k:60s}  shape={str(list(tensor.shape)):20s}  dtype={tensor.dtype}")

        # Also probe base counterparts of shared weights
        print(f"\nShared weight counterparts:")
        for mtp_k, base_k in shared_candidates:
            if base_k in weight_map:
                fpath = os.path.join(model_dir, weight_map[base_k])
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(base_k)
                    print(f"  {base_k:60s}  shape={str(list(tensor.shape)):20s}  dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
