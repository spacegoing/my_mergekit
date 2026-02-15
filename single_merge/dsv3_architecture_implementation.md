# DeepSeek-V3 Architecture Support for mergekit

## Problem

mergekit had no architecture definition for `DeepseekV3ForCausalLM`. Without
it, auto-inference used layer 0 (dense MLP) as the template for all 41
layers, marking `mlp.up_proj.weight` as required. MoE layers (1-39) don't
have this weight — they have `mlp.experts.X.up_proj.weight` instead — causing
the merge to crash:

```
RuntimeError: Tensor model.layers.35.mlp.up_proj.weight required but not
present in model ...
```

## Solution

Three files added/modified, following the GLM4 MoE pattern
(`mergekit/architecture/moe_defs.py:150-206`, commit `8146a66`).

---

## Files

### 1. `mergekit/_data/architectures/deepseekv3.json` (new)

Defines the "dense layer" weight template — the 12 tensor names that appear
in layer 0 (MLA attention + dense MLP + layernorms).

### 2. `mergekit/architecture/moe_defs.py` (modified)

Added `DeepseekV3ModuleArchitecture` class that routes each layer to the
correct weight template.

### 3. `mergekit/architecture/__init__.py` (modified)

Registered the new class in the architecture dispatcher.

---

## Evidence: Every Fact Traced to Source

### Dense layer structure (12 weights) — `deepseekv3.json`

**Source**: `ckpt_inspect_results.text` lines 93-105, "Base layer 0 structure
(12 keys)":

| # | JSON template | Checkpoint evidence |
|---|---------------|---------------------|
| 1 | `model.layers.${layer_index}.input_layernorm.weight` | line 94: `.input_layernorm.weight` |
| 2 | `model.layers.${layer_index}.mlp.down_proj.weight` | line 95: `.mlp.down_proj.weight` |
| 3 | `model.layers.${layer_index}.mlp.gate_proj.weight` | line 96: `.mlp.gate_proj.weight` |
| 4 | `model.layers.${layer_index}.mlp.up_proj.weight` | line 97: `.mlp.up_proj.weight` |
| 5 | `model.layers.${layer_index}.post_attention_layernorm.weight` | line 98: `.post_attention_layernorm.weight` |
| 6 | `model.layers.${layer_index}.self_attn.kv_a_layernorm.weight` | line 99: `.self_attn.kv_a_layernorm.weight` |
| 7 | `model.layers.${layer_index}.self_attn.kv_a_proj_with_mqa.weight` | line 100: `.self_attn.kv_a_proj_with_mqa.weight` |
| 8 | `model.layers.${layer_index}.self_attn.kv_b_proj.weight` | line 101: `.self_attn.kv_b_proj.weight` |
| 9 | `model.layers.${layer_index}.self_attn.o_proj.weight` | line 102: `.self_attn.o_proj.weight` |
| 10 | `model.layers.${layer_index}.self_attn.q_a_layernorm.weight` | line 103: `.self_attn.q_a_layernorm.weight` |
| 11 | `model.layers.${layer_index}.self_attn.q_a_proj.weight` | line 104: `.self_attn.q_a_proj.weight` |
| 12 | `model.layers.${layer_index}.self_attn.q_b_proj.weight` | line 105: `.self_attn.q_b_proj.weight` |

### Pre/post weights — `deepseekv3.json`

**Source**: `ckpt_inspect_results.text` lines 62-67, "Weight categories":

| JSON | Checkpoint evidence |
|------|---------------------|
| `model.embed_tokens.weight` (is_embed=true) | line 63: `embedding: 1 keys` |
| `model.norm.weight` | line 64: `final_norm: 1 keys` |
| `lm_head.weight` (is_embed=true) | line 65: `lm_head: 1 keys` |

### MoE layer structure (782 weights) — `moe_defs.py`

**Source**: `ckpt_inspect_results.text` lines 107-893, "Last base layer 39
structure (782 keys)":

| Component | Count | Code line(s) | Checkpoint evidence |
|-----------|-------|--------------|---------------------|
| `mlp.experts.{0-255}.{gate,up,down}_proj.weight` | 256 x 3 = 768 | moe_defs.py:209-218 | lines 109-633: experts 0-255, 3 params each |
| `mlp.gate.weight` | 1 | moe_defs.py:219 | line 878: `.mlp.gate.weight` |
| `mlp.gate.e_score_correction_bias` | 1 | moe_defs.py:220 | line 877: `.mlp.gate.e_score_correction_bias` |
| `mlp.shared_experts.{gate,up,down}_proj.weight` | 3 | moe_defs.py:225-229 | lines 879-881: `.mlp.shared_experts.{gate,up,down}_proj.weight` |
| Non-MLP (attention + layernorms) | 9 | moe_defs.py:230-233 | lines 882-890: same 9 as layer 0 minus 3 MLP |

Total: 768 + 1 + 1 + 3 + 9 = **782** (matches line 107: "782 keys")

### MTP layer structure (16 weights) — `moe_defs.py`

**Source**: `ckpt_inspect_results.text` lines 73-91, "MTP layer keys (16)":

| Component | Count | Code line(s) | Checkpoint evidence |
|-----------|-------|--------------|---------------------|
| Dense template (attention + MLP + norms) | 12 | moe_defs.py:186 | lines 79-90: same 12 as layer 0 |
| `eh_proj.weight` | 1 | moe_defs.py:188 | line 76: `model.layers.40.eh_proj.weight` |
| `enorm.weight` | 1 | moe_defs.py:189 | line 77: `model.layers.40.enorm.weight` |
| `hnorm.weight` | 1 | moe_defs.py:190 | line 78: `model.layers.40.hnorm.weight` |
| `shared_head.norm.weight` | 1 | moe_defs.py:191 | line 91: `model.layers.40.shared_head.norm.weight` |

Total: 12 + 4 = **16** (matches line 67: "mtp_layer: 16 keys")

### Config field names — `moe_defs.py`

Every config field access is verified against `dsv3_40B_info.text`:

| Code | Config field | Value | Evidence |
|------|-------------|-------|----------|
| `config.architectures[0]` | `architectures` | `["DeepseekV3ForCausalLM"]` | dsv3_40B_info.text line 3-5 |
| `config.n_routed_experts` | `n_routed_experts` | `256` | dsv3_40B_info.text line 27 |
| `config.num_hidden_layers` | `num_hidden_layers` | `40` | dsv3_40B_info.text line 32 |
| `config.num_nextn_predict_layers` | `num_nextn_predict_layers` | `1` | dsv3_40B_info.text line 34 |
| `config.first_k_dense_replace` | `first_k_dense_replace` | `1` | dsv3_40B_info.text line 16 |
| `config.moe_layer_freq` | `moe_layer_freq` | `1` | dsv3_40B_info.text line 25 |
| `config.n_shared_experts` | `n_shared_experts` | `1` | dsv3_40B_info.text line 28 |
| `config.model_type` | `model_type` | `"deepseek_v3"` | dsv3_40B_info.text line 23 |

### JSON format — `deepseekv3.json`

Follows the mergekit JSON architecture schema used by all other architecture
definitions. Pattern copied from `mergekit/_data/architectures/glm4moe.json`.

| Field | Purpose | Reference |
|-------|---------|-----------|
| `model_type` | Matches config.model_type | glm4moe.json line 2 |
| `architectures` | Matches config.architectures | glm4moe.json line 3-5 |
| `pre_weights` | Weights before layers | glm4moe.json line 6-10 |
| `num_layers_config_key` | Config key for layer count | glm4moe.json line 12 |
| `layer_templates.weights` | Per-layer weight patterns | glm4moe.json line 13-62 |
| `post_weights` | Weights after layers | glm4moe.json line 64-76 |

### Python class pattern — `moe_defs.py`

`DeepseekV3ModuleArchitecture` follows `Glm4MoeModuleArchitecture`
(moe_defs.py:150-206) exactly:

| Pattern | GLM4 | DeepSeek-V3 |
|---------|------|-------------|
| Load JSON base | `GLM4_INFO = NAME_TO_ARCH["Glm4MoeForCausalLM"][0]` | `DSV3_INFO = NAME_TO_ARCH["DeepseekV3ForCausalLM"][0]` |
| Delegate pre/post | `GLM4_MODULE_ARCH.pre_weights(config)` | `DSV3_MODULE_ARCH.pre_weights(config)` |
| Dense layer routing | `index < config.first_k_dense_replace` | Same |
| MoE expert loop | `range(self.num_experts)` | Same |
| Gate weight | `prefix + ".mlp.gate.weight"` | Same |
| Score correction | `prefix + ".mlp.gate.e_score_correction_bias"` | Same |
| Shared experts | Hardcoded 3 params | Conditional on `n_shared_experts > 0` |
| Non-MLP filtering | `if ".mlp." in weight_info.name: continue` | Same |

**Additions beyond GLM4 pattern:**

| Addition | Reason | Evidence |
|----------|--------|----------|
| `num_layers()` override: returns `num_hidden + num_mtp` | DeepSeek-V3 has MTP layer at index `num_hidden_layers` | ckpt_inspect line 69: "Layer indices present: 0..40" |
| `num_layers_config_key()` returns `None` | No single config key = total layers (total = `num_hidden_layers + num_nextn_predict_layers`). Prevents `_model_out_config()` from overwriting `num_hidden_layers` with slice count, which would break the MTP/MoE layer boundary. | See "Output Config Bug" section below |
| MTP branch in `layer_weights()` | Layer 40 has unique weights (eh_proj, enorm, hnorm, shared_head.norm) | ckpt_inspect lines 76-91 |
| `moe_layer_freq` check | Not all layers after `first_k_dense_replace` are guaranteed MoE | dsv3_40B_info.text line 25: `"moe_layer_freq": 1` |
| `n_shared_experts` conditional | Shared experts may not exist in all variants | dsv3_40B_info.text line 28: `"n_shared_experts": 1` |

### Registration — `__init__.py`

Follows the same `elif` dispatch pattern as Mixtral, Qwen3, Afmoe, GLM4
(lines 42-76). Returns `ModelArchitecture` with `model_type="deepseek_v3"`.

---

## Verification: Total Weight Count

```
pre_weights:    1  (model.embed_tokens.weight)
post_weights:   2  (model.norm.weight, lm_head.weight)
layer 0:       12  (dense) x 1 layer  =    12
layers 1-39:  782  (MoE)   x 39 layers = 30498
layer 40:      16  (MTP)   x 1 layer   =    16
                                        ------
Grand total:                             30529
```

**Checkpoint**: `ckpt_inspect_results.text` line 41: `total keys: 30529` --
exact match.

**Category breakdown**:
- `embedding: 1` = pre_weights (1) -- match
- `final_norm: 1` + `lm_head: 1` = post_weights (2) -- match
- `base_layer: 30510` = 1 x 12 + 39 x 782 = 30510 -- match
- `mtp_layer: 16` = 1 x 16 = 16 -- match

---

## Output Config Bug (Fixed)

### Problem

`merge.py:_model_out_config()` computes total output layers from slices and
sets `num_hidden_layers` to that total:

```
Slice 0: layers [0, 40) → 40 layers
Slice 1: layers [40, 41) → 1 layer
Total: 41 → set_config_value(cfg_out, "num_hidden_layers", 41)
```

With `cfg_out.num_hidden_layers = 41`, `layer_weights(40, cfg_out)` checks
`40 >= 41` → False → treats layer 40 as MoE (782 weights). But the base
model's config has `num_hidden_layers = 40`, so `layer_weights(40, base_config)`
correctly identifies it as MTP (16 weights).

**Result**: `plan_layer()` has `weights_out` = 782, `weights_in[0]` = 16 →
`IndexError: list index out of range` at `plan.py:260`.

### Fix

`num_layers_config_key()` returns `None`. This causes `_model_out_config()` to
skip the `set_config_value` call (handled at merge.py line 323-329: `if not
cfg_key: continue`). The output config keeps `num_hidden_layers = 40` from the
base model, preserving the correct MTP/MoE boundary.

This is safe because:
1. `num_layers()` is overridden and does NOT call `num_layers_config_key()`
2. `normalize_config()` uses `num_layers()` directly (not the config key)
3. For full merges, the base model config already has correct values
4. `JsonModuleArchDef.num_layers_config_key` is `Optional[str]` — `None` is
   an established pattern in mergekit's architecture system

---

## RL Checkpoint Compatibility

The RL checkpoint (`ckpt_inspect_results.text` lines 897-962) has
`num_nextn_predict_layers: 0` but **layer 40 weights are physically present**
(16 MTP keys). This works because:

1. Both checkpoints produce `DeepseekV3ModuleArchitecture(num_experts=256)` --
   Pydantic equality passes (only `num_experts` is stored as a field)
2. `num_layers()` reads `num_nextn_predict_layers` from config at runtime,
   not construction time -- the architecture object is shared
3. `get_architecture_info()` returns the base model's architecture (41 layers)
4. Slices in `dare_single.py` only reference RL for layers 0-39, never
   layer 40 -- so mergekit never tries to determine RL's MTP layer count
