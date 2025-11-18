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

"""Provide model customization plugins for v1.

This module contains plugins for model optimization, including hardware-specific
kernel replacements, tokenizer extensions, and parameter-efficient fine-tuning
adapters. These plugins enable performance optimization and efficient adaptation
of language models.

Key Submodules:
    kernels: Hardware-optimized kernel implementations for NPU, CUDA, and other
        accelerators, including RMSNorm, SwiGLU, RoPE, and FlashAttention.
    added_token: Tokenizer extensions for adding special tokens.
    peft: Parameter-efficient fine-tuning adapters (LoRA, etc.).

Example:
    Apply NPU-optimized kernels to a model::

        from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
        from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import NpuRMSNormKernel

        model = apply_kernel(model, NpuRMSNormKernel)

See Also:
    llamafactory.v1.plugins.model_plugins.kernels: Kernel implementations.
    llamafactory.v1.core.model_engine: Uses model plugins during loading.
"""
