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

"""Apply Liger kernel optimizations for faster training.

This module integrates the Liger kernel library which provides optimized CUDA
kernels for transformer operations. Liger kernels can significantly speed up
training by fusing operations and reducing memory access overhead.

Key Functions:
    apply_liger_kernel: Apply Liger kernel optimizations based on model type.

Supported Model Types:
    - Gemma (1, 2, 3, 3_text)
    - GLM4, GLM4V
    - Granite
    - LLaMA, LLaVA, Mistral, Mixtral, MLLaMA
    - OLMo2
    - PaliGemma
    - Phi3
    - Qwen2, Qwen2-VL, Qwen2.5-VL, Qwen3, Qwen3-MoE

The module automatically handles chunked cross-entropy vs regular cross-entropy
based on the training stage (PT/SFT vs RM/PPO/DPO).

Example:
    >>> from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
    >>> from transformers import AutoConfig
    >>>
    >>> config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> model_args.enable_liger_kernel = True
    >>> apply_liger_kernel(
    ...     config, model_args,
    ...     is_trainable=True,
    ...     require_logits=False  # PT/SFT stage
    ... )
    >>> # Liger kernels are now applied to the model

See Also:
    llamafactory.model.loader: Calls apply_liger_kernel before model loading.
    https://github.com/linkedin/Liger-Kernel: Liger kernel documentation.
"""

import inspect
from typing import TYPE_CHECKING

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    if model_type == "gemma":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_liger_kernel
    elif model_type == "gemma2":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_liger_kernel
    elif model_type == "gemma3":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3 as apply_liger_kernel
    elif model_type == "gemma3_text":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text as apply_liger_kernel
    elif model_type == "glm4":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4 as apply_liger_kernel
    elif model_type == "glm4v":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v as apply_liger_kernel
    elif model_type == "granite":
        from liger_kernel.transformers import apply_liger_kernel_to_granite as apply_liger_kernel
    elif model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel
    elif model_type == "llava":
        from liger_kernel.transformers import apply_liger_kernel_to_llava as apply_liger_kernel
    elif model_type == "mistral":
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_liger_kernel
    elif model_type == "mixtral":
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_liger_kernel
    elif model_type == "mllama":
        from liger_kernel.transformers import apply_liger_kernel_to_mllama as apply_liger_kernel
    elif model_type == "olmo2":
        from liger_kernel.transformers import apply_liger_kernel_to_olmo2 as apply_liger_kernel
    elif model_type == "paligemma":
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma as apply_liger_kernel
    elif model_type == "phi3":
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_liger_kernel
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_liger_kernel
    elif model_type == "qwen2_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl as apply_liger_kernel
    elif model_type == "qwen2_5_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl as apply_liger_kernel
    elif model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_liger_kernel
    else:
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
        logger.info_rank0("Current training stage does not support chunked cross entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}

    apply_liger_kernel(**kwargs)
    logger.info_rank0("Liger kernel has been applied to the model.")
