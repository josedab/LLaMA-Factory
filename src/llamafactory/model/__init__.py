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

"""Provide model loading and configuration utilities for LLM fine-tuning.

This module serves as the main entry point for model-related operations in
LlamaFactory. It exposes essential functions for loading pre-trained models,
tokenizers, and configurations, as well as utilities for model quantization
and parameter analysis.

Key Functions:
    load_model: Load a pre-trained model with optional adapters and quantization.
    load_tokenizer: Load a tokenizer and optional processor for a model.
    load_config: Load model configuration from a checkpoint.
    find_all_linear_modules: Find all linear modules suitable for LoRA/GaLore.
    load_valuehead_params: Load value head parameters for RLHF training.

Key Classes:
    QuantizationMethod: Enum for supported quantization methods (BNB, GPTQ, etc.).

Example:
    >>> from llamafactory.model import load_tokenizer, load_model
    >>> from llamafactory.hparams import ModelArguments, FinetuningArguments
    >>> model_args = ModelArguments(model_name_or_path="meta-llama/Llama-2-7b-hf")
    >>> finetuning_args = FinetuningArguments(finetuning_type="lora")
    >>> tokenizer_module = load_tokenizer(model_args)
    >>> model = load_model(
    ...     tokenizer_module["tokenizer"],
    ...     model_args,
    ...     finetuning_args,
    ...     is_trainable=True
    ... )

See Also:
    llamafactory.model.loader: Detailed model loading implementation.
    llamafactory.model.adapter: Adapter initialization for fine-tuning.
    llamafactory.model.patcher: Model patching and modifications.
"""

from .loader import load_config, load_model, load_tokenizer
from .model_utils.misc import find_all_linear_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.valuehead import load_valuehead_params


__all__ = [
    "QuantizationMethod",
    "find_all_linear_modules",
    "load_config",
    "load_model",
    "load_tokenizer",
    "load_valuehead_params",
]
