# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provide miscellaneous model utilities for module discovery and registration.

This module contains utility functions for analyzing model architecture,
finding modules suitable for adaptation methods (LoRA, GaLore, APOLLO),
and registering models with HuggingFace's Auto classes.

Key Functions:
    find_all_linear_modules: Find linear modules for LoRA/GaLore application.
    find_expanded_modules: Find modules in expanded blocks for LLaMA-Pro.
    register_autoclass: Register model components for Auto class loading.

Module Discovery:
    find_all_linear_modules discovers all Linear layers while excluding:
    - Output layers (lm_head, output_layer, output)
    - Projector modules for composite models
    - Vision tower modules (when freeze_vision_tower is True)

LLaMA-Pro Support:
    find_expanded_modules identifies modules in specific layers for the
    LLaMA-Pro training strategy where only expanded layers are trained.

Example:
    >>> from llamafactory.model.model_utils.misc import find_all_linear_modules
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> linear_modules = find_all_linear_modules(model, freeze_vision_tower=False)
    >>> print(linear_modules)
    >>> # ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

See Also:
    llamafactory.model.adapter: Uses find_all_linear_modules for LoRA target selection.
    llamafactory.model.model_utils.visual: COMPOSITE_MODELS registry used here.
"""

from typing import TYPE_CHECKING

from ...extras import logging
from .visual import COMPOSITE_MODELS


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> list[str]:
    r"""Find all available modules to apply LoRA, GaLore or APOLLO."""
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")

    if model_type in COMPOSITE_MODELS:
        forbidden_modules.add(COMPOSITE_MODELS[model_type].projector_key)

    if freeze_vision_tower and model_type in COMPOSITE_MODELS:
        forbidden_modules.update(COMPOSITE_MODELS[model_type].vision_model_keys)

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def find_expanded_modules(model: "PreTrainedModel", target_modules: list[str], num_layer_trainable: int) -> list[str]:
    r"""Find the modules in the expanded blocks to apply lora."""
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("Model was not supported.")

    if num_layers % num_layer_trainable != 0:
        raise ValueError(
            f"`num_layers` {num_layers} should be divisible by `num_layer_trainable` {num_layer_trainable}."
        )

    stride = num_layers // num_layer_trainable
    trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    trainable_layers = [f".{idx:d}." for idx in trainable_layer_ids]
    module_names = []
    for name, _ in model.named_modules():
        if any(target_module in name for target_module in target_modules) and any(
            trainable_layer in name for trainable_layer in trainable_layers
        ):
            module_names.append(name)

    logger.info_rank0("Apply lora to layers: {}.".format(",".join(map(str, trainable_layer_ids))))
    return module_names


def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
