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

"""Configure key-value cache settings for model inference and training.

This module manages the KV cache configuration for transformer models. The
KV cache stores computed key-value pairs from attention layers to avoid
redundant computation during autoregressive generation.

Key Functions:
    configure_kv_cache: Set KV cache usage based on training/inference mode.

Behavior:
    - Training: KV cache is always disabled (incompatible with backpropagation).
    - Inference: KV cache is enabled by default for faster generation.

The function handles both standard models and composite (multimodal) models
that have separate text_config configurations.

Example:
    >>> from llamafactory.model.model_utils.kv_cache import configure_kv_cache
    >>> from transformers import AutoConfig
    >>>
    >>> config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>>
    >>> # For inference - enable KV cache
    >>> model_args.use_kv_cache = True
    >>> configure_kv_cache(config, model_args, is_trainable=False)
    >>> # config.use_cache is now True
    >>>
    >>> # For training - always disabled
    >>> configure_kv_cache(config, model_args, is_trainable=True)
    >>> # config.use_cache is now False

See Also:
    llamafactory.model.patcher: Calls configure_kv_cache in patch_config.
    llamafactory.hparams.ModelArguments: Contains use_kv_cache setting.
"""

from typing import TYPE_CHECKING

from ...extras import logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


def configure_kv_cache(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable:
        setattr(config, "use_cache", model_args.use_kv_cache)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", model_args.use_kv_cache)

        if model_args.use_kv_cache:
            logger.info_rank0("KV cache is enabled for faster generation.")
        else:
            logger.info_rank0("KV cache is disabled.")
    else:
        setattr(config, "use_cache", False)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", False)

        logger.info_rank0("KV cache is disabled during training.")
