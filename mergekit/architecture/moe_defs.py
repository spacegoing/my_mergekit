# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import ClassVar, List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModuleArchitecture,
    WeightInfo,
)
from mergekit.architecture.json_definitions import NAME_TO_ARCH

MISTRAL_INFO = NAME_TO_ARCH["MistralForCausalLM"][0]
MISTRAL_MODULE_ARCH = MISTRAL_INFO.modules["default"].architecture


class MixtralModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "mixtral"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralModuleArchitecture(num_local_experts=config.num_local_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return MISTRAL_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_experts = self.num_local_experts
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                tensor_names.append(
                    prefix + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".block_sparse_moe.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in MISTRAL_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res


QWEN3_INFO = NAME_TO_ARCH["Qwen3ForCausalLM"][0]
QWEN3_MODULE_ARCH = QWEN3_INFO.modules["default"].architecture


class Qwen3MoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "Qwen3MoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "qwen3_moe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return Qwen3MoeModuleArchitecture(num_experts=config.num_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return QWEN3_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return QWEN3_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return QWEN3_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(self.num_experts):
            for param in ("up_proj", "gate_proj", "down_proj"):
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".mlp.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in QWEN3_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res


AFMOE_PARTIAL_INFO = NAME_TO_ARCH["_AfmoePartialForCausalLM"][0]
AFMOE_PARTIAL_MODULE_ARCH = AFMOE_PARTIAL_INFO.modules["default"].architecture


class AfmoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "AfmoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "afmoe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return AfmoeModuleArchitecture(num_experts=config.num_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return AFMOE_PARTIAL_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return AFMOE_PARTIAL_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return AFMOE_PARTIAL_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        res = AFMOE_PARTIAL_MODULE_ARCH.layer_weights(index, config) or []
        prefix = f"model.layers.{index}"
        for expert_idx in range(self.num_experts):
            for param in ("up_proj", "gate_proj", "down_proj"):
                res.append(
                    WeightInfo(
                        name=prefix + f".mlp.experts.{expert_idx}.{param}.weight",
                        optional=True,
                    )
                )
        return res


DSV3_INFO = NAME_TO_ARCH["DeepseekV3ForCausalLM"][0]
DSV3_MODULE_ARCH = DSV3_INFO.modules["default"].architecture


class DeepseekV3ModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "DeepseekV3ForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "deepseek_v3"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return DeepseekV3ModuleArchitecture(
            num_experts=config.n_routed_experts,
        )

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return DSV3_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return DSV3_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        # Return None: total layers = num_hidden_layers + num_nextn_predict_layers,
        # so no single config key represents the total. Returning None prevents
        # _model_out_config from overwriting num_hidden_layers with the total
        # slice count (which would break the MTP vs MoE layer boundary).
        return None

    def num_layers(self, config: PretrainedConfig) -> int:
        num_hidden = config.num_hidden_layers
        num_mtp = getattr(config, "num_nextn_predict_layers", 0)
        return num_hidden + num_mtp

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_hidden = config.num_hidden_layers
        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        prefix = f"model.layers.{index}"

        if index >= num_hidden:
            # MTP layer: dense attention + dense MLP + MTP-specific weights
            res = list(DSV3_MODULE_ARCH.layer_weights(index, config))
            for mtp_name in (
                "eh_proj.weight",
                "enorm.weight",
                "hnorm.weight",
                "shared_head.norm.weight",
            ):
                res.append(WeightInfo(name=prefix + "." + mtp_name))
            return res
        elif index < first_k_dense:
            # Dense layer: use base JSON definition as-is
            return DSV3_MODULE_ARCH.layer_weights(index, config)
        else:
            # Check if this layer is MoE based on moe_layer_freq
            moe_layer_freq = getattr(config, "moe_layer_freq", 1)
            is_moe = (index - first_k_dense) % moe_layer_freq == 0

            if not is_moe:
                # Interleaved dense layer within MoE range
                return DSV3_MODULE_ARCH.layer_weights(index, config)

            # MoE layer: routed experts + shared experts + gate + non-MLP base
            tensor_names = []
            for expert_idx in range(self.num_experts):
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.gate_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.up_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.down_proj.weight"
                )
            tensor_names.append(prefix + ".mlp.gate.weight")
            tensor_names.append(prefix + ".mlp.gate.e_score_correction_bias")
            res = []
            for name in tensor_names:
                res.append(WeightInfo(name=name))
            n_shared = getattr(config, "n_shared_experts", 0)
            if n_shared > 0:
                for param in ("gate_proj", "up_proj", "down_proj"):
                    res.append(WeightInfo(
                        name=prefix + f".mlp.shared_experts.{param}.weight",
                    ))
            for weight_info in DSV3_MODULE_ARCH.layer_weights(index, config):
                if ".mlp." in weight_info.name:
                    continue
                res.append(weight_info)
            return res


GLM4_INFO = NAME_TO_ARCH["Glm4MoeForCausalLM"][0]
GLM4_MODULE_ARCH = GLM4_INFO.modules["default"].architecture


class Glm4MoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "Glm4MoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "glm4_moe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return Glm4MoeModuleArchitecture(num_experts=config.n_routed_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return GLM4_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return GLM4_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return GLM4_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"

        if index < config.first_k_dense_replace:
            return GLM4_MODULE_ARCH.layer_weights(index, config)
        else:
            tensor_names = []
            for expert_idx in range(self.num_experts):
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.gate_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.up_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.down_proj.weight"
                )
            tensor_names.append(prefix + ".mlp.gate.weight")
            tensor_names.append(prefix + ".mlp.gate.e_score_correction_bias")
            shared_expert_names = [
                (prefix + ".mlp.shared_experts.gate_proj.weight", False),
                (prefix + ".mlp.shared_experts.up_proj.weight", False),
                (prefix + ".mlp.shared_experts.down_proj.weight", False),
            ]
            res = []
            for name in tensor_names:
                res.append(WeightInfo(name=name))
            for name, optional in shared_expert_names:
                res.append(WeightInfo(name=name, optional=optional))
            for weight_info in GLM4_MODULE_ARCH.layer_weights(index, config):
                if ".mlp." in weight_info.name:
                    continue
                res.append(weight_info)
            return res
