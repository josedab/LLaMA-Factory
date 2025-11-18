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

"""Support Mixture-of-Depths (MoD) model loading and conversion.

This module provides integration with the MoD library for loading and
converting models with Mixture-of-Depths architecture. MoD allows models
to dynamically skip computation for certain tokens, improving efficiency.

Key Functions:
    load_mod_pretrained_model: Load a pre-trained MoD model.
    convert_pretrained_model_to_mod: Convert standard model to MoD architecture.

Mixture-of-Depths Concept:
    Unlike standard transformers where every token passes through all layers,
    MoD models learn to route tokens dynamically, skipping layers for tokens
    that don't need full processing. This can significantly reduce computation
    while maintaining quality.

Supported Models:
    Models listed in MOD_SUPPORTED_MODELS constant (typically LLaMA variants).

Usage Modes:
    - "load": Load an existing MoD checkpoint.
    - "convert": Convert a standard model to MoD architecture.

Example:
    >>> from llamafactory.model.model_utils.mod import convert_pretrained_model_to_mod
    >>> from transformers import AutoModelForCausalLM, AutoConfig
    >>>
    >>> config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> model_args.mixture_of_depths = "convert"
    >>> mod_model = convert_pretrained_model_to_mod(model, config, model_args)

See Also:
    llamafactory.model.loader: Uses MoD functions based on mixture_of_depths arg.
    https://arxiv.org/abs/2404.02258: Mixture-of-Depths paper.
"""

from typing import TYPE_CHECKING

from ...extras.constants import MOD_SUPPORTED_MODELS


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def load_mod_pretrained_model(**init_kwargs) -> "PreTrainedModel":
    from MoD import AutoMoDModelForCausalLM

    return AutoMoDModelForCausalLM.from_pretrained(**init_kwargs)


def convert_pretrained_model_to_mod(
    model: "PreTrainedModel", config: "PretrainedConfig", model_args: "ModelArguments"
) -> "PreTrainedModel":
    from MoD import apply_mod_to_hf

    if getattr(config, "model_type", None) not in MOD_SUPPORTED_MODELS:
        raise ValueError("Current model is not supported by mixture-of-depth.")

    model = apply_mod_to_hf(model)
    model = model.to(model_args.compute_dtype)
    return model
