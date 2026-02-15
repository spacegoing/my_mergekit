#!/usr/bin/env python3
"""Merge MTP (Multi-Token Prediction) layer weights from an original DeepSeek-V3
checkpoint into an exported checkpoint that was trained with mtp_num_layers=null.

The original checkpoint has layer N (the MTP layer, where N = num_hidden_layers)
but the exported checkpoint does not. This script:
  1. Reads config.json to determine MTP layer index = num_hidden_layers.
  2. Copies all 16 MTP weight keys from the original checkpoint.
     (These were never trained — MTP was disabled during FAPO.)
  3. Saves them as a new safetensors shard in the output directory.
  4. Updates model.safetensors.index.json with the new shard.
  5. Ensures num_nextn_predict_layers is set in config.json.

Note: Shared weights (embed_tokens, shared_head.head) are NOT stored as separate
keys in the checkpoint for this 40-layer model variant. The HF modeling code
handles the sharing internally, so we only need to copy the 16 unique keys.

Usage:
    python merge_mtp_layer.py \\
        --original /path/to/original/model \\
        --exported /path/to/exported/model \\
        [--output   /path/to/merged/output]   # default: <exported>_with_mtp
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file


# Expected 16 MTP layer weight suffixes (dense MLP + MLA attention + MTP-specific)
EXPECTED_MTP_SUFFIXES = {
    # MTP-specific
    "enorm.weight",
    "hnorm.weight",
    "eh_proj.weight",
    "shared_head.norm.weight",
    # Attention (MLA)
    "input_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    # Dense MLP
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def load_index(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No model.safetensors.index.json found in {model_dir}")
    with open(index_path) as f:
        return json.load(f)


def get_mtp_keys(index: dict, mtp_layer_idx: int) -> list[str]:
    """Extract all weight keys belonging to MTP layer from index."""
    prefix = f"model.layers.{mtp_layer_idx}."
    return sorted(k for k in index["weight_map"] if k.startswith(prefix))


def load_tensors(model_dir: str, keys: list[str], index: dict) -> dict:
    """Load specific tensors from a safetensors model directory."""
    file_to_keys = defaultdict(list)
    weight_map = index["weight_map"]
    for key in keys:
        file_to_keys[weight_map[key]].append(key)

    tensors = {}
    for fname, key_list in file_to_keys.items():
        filepath = os.path.join(model_dir, fname)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in key_list:
                tensors[key] = f.get_tensor(key)
    return tensors


def main():
    parser = argparse.ArgumentParser(
        description="Merge MTP layer from original checkpoint into exported checkpoint"
    )
    parser.add_argument(
        "--original", required=True,
        help="Path to original model WITH MTP layer (e.g. the DPO base model)"
    )
    parser.add_argument(
        "--exported", required=True,
        help="Path to exported model WITHOUT MTP layer (FAPO training output)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: <exported>_with_mtp)"
    )
    args = parser.parse_args()

    output_dir = args.output or f"{args.exported.rstrip('/')}_with_mtp"

    print(f"Original model (with MTP): {args.original}")
    print(f"Exported model (no MTP):   {args.exported}")
    print(f"Output model:              {output_dir}")

    # ── Read configs ──────────────────────────────────────────────────────
    orig_config = load_config(args.original)
    num_hidden = orig_config["num_hidden_layers"]
    mtp_layer_idx = num_hidden  # MTP layer sits right after base layers
    num_mtp = orig_config.get("num_nextn_predict_layers", 0)

    print(f"\nConfig: num_hidden_layers={num_hidden}, num_nextn_predict_layers={num_mtp}")
    print(f"  => MTP layer index: {mtp_layer_idx}")

    if num_mtp != 1:
        raise ValueError(f"Expected num_nextn_predict_layers=1, got {num_mtp}")

    # ── Load indices ──────────────────────────────────────────────────────
    orig_index = load_index(args.original)
    export_index = load_index(args.exported)

    # ── Find MTP keys in original ─────────────────────────────────────────
    mtp_keys = get_mtp_keys(orig_index, mtp_layer_idx)
    if not mtp_keys:
        raise ValueError(
            f"No MTP layer keys (model.layers.{mtp_layer_idx}.*) found in "
            f"original model at {args.original}"
        )

    print(f"\nMTP layer: {len(mtp_keys)} keys in original")

    # Validate structure matches expectation
    prefix = f"model.layers.{mtp_layer_idx}."
    actual_suffixes = {k[len(prefix):] for k in mtp_keys}
    if actual_suffixes != EXPECTED_MTP_SUFFIXES:
        missing = EXPECTED_MTP_SUFFIXES - actual_suffixes
        extra = actual_suffixes - EXPECTED_MTP_SUFFIXES
        if missing:
            print(f"  WARNING: missing expected suffixes: {missing}")
        if extra:
            print(f"  WARNING: unexpected extra suffixes: {extra}")
    else:
        print(f"  Structure matches expected 16 MTP keys. OK.")

    # ── Verify no MTP keys in exported ────────────────────────────────────
    existing_mtp = get_mtp_keys(export_index, mtp_layer_idx)
    if existing_mtp:
        print(f"\n  WARNING: exported model already has {len(existing_mtp)} "
              f"layer-{mtp_layer_idx} keys! They will be overwritten.")

    # ── Load MTP weights from original ────────────────────────────────────
    print(f"\nLoading MTP weights from original...")
    mtp_tensors = load_tensors(args.original, mtp_keys, orig_index)

    total_params = sum(t.numel() for t in mtp_tensors.values())
    total_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"  {len(mtp_tensors)} tensors, {total_params:,} params ({total_bytes / 1e9:.3f} GB)")

    for k in sorted(mtp_tensors):
        t = mtp_tensors[k]
        print(f"    {k:60s} {str(list(t.shape)):20s} {t.dtype}")

    # ── Copy exported model to output ─────────────────────────────────────
    if os.path.abspath(output_dir) != os.path.abspath(args.exported):
        print(f"\nCopying exported model to {output_dir}...")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(args.exported, output_dir)
    else:
        print("\nModifying exported model in-place...")

    # ── Save MTP weights as new safetensors shard ─────────────────────────
    mtp_shard_name = "model-mtp-layer.safetensors"
    mtp_shard_path = os.path.join(output_dir, mtp_shard_name)

    print(f"\nSaving MTP weights to {mtp_shard_name}...")
    save_file(mtp_tensors, mtp_shard_path)
    saved_size = os.path.getsize(mtp_shard_path)
    print(f"  Saved: {saved_size / 1e9:.3f} GB")

    # ── Update index.json ─────────────────────────────────────────────────
    updated_index = json.loads(json.dumps(export_index))  # deep copy

    for key in mtp_tensors:
        updated_index["weight_map"][key] = mtp_shard_name

    if "total_size" in updated_index.get("metadata", {}):
        updated_index["metadata"]["total_size"] += total_bytes

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(updated_index, f, indent=2)
    print(f"Updated {index_path}")
    print(f"  Total keys in index: {len(updated_index['weight_map'])}")

    # ── Update config.json ────────────────────────────────────────────────
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        old_val = config.get("num_nextn_predict_layers", "NOT SET")
        config["num_nextn_predict_layers"] = num_mtp
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nconfig.json: num_nextn_predict_layers = {old_val} -> {num_mtp}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE. Merged model saved to: {output_dir}")
    print(f"  MTP layer {mtp_layer_idx}: {len(mtp_tensors)} tensors ({total_bytes / 1e9:.3f} GB)")
    print(f"  Verify with: python inspect_mtp_ckpt.py {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
