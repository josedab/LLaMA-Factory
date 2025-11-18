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

"""Provide model utility functions for configuration and optimization.

This subpackage contains specialized utilities for various aspects of model
configuration, optimization, and patching. Each module handles a specific
concern such as attention mechanisms, quantization, or visual model support.

Modules:
    attention: Configure attention implementations (FA2, SDPA, eager).
    checkpointing: Gradient checkpointing and training preparation.
    embedding: Token embedding resizing and initialization.
    ktransformers: KTransformers integration for efficient inference.
    kv_cache: Key-value cache configuration.
    liger_kernel: Liger kernel optimizations for faster training.
    longlora: LongLoRA shifted sparse attention implementation.
    misc: Miscellaneous utilities (linear module finding, autoclass registration).
    mod: Mixture-of-Depths model support.
    moe: Mixture-of-Experts configuration and DeepSpeed Z3 leaf modules.
    packing: Sequence packing with block diagonal attention.
    quantization: Model quantization (BNB, GPTQ, HQQ, EETQ).
    rope: RoPE scaling configuration (linear, dynamic, yarn, llama3).
    unsloth: Unsloth integration for optimized training.
    valuehead: Value head loading for RLHF.
    visual: Vision-language model support and composite model registry.

Example:
    >>> from llamafactory.model.model_utils.quantization import configure_quantization
    >>> from llamafactory.model.model_utils.attention import configure_attn_implementation
    >>> from llamafactory.model.model_utils.misc import find_all_linear_modules

See Also:
    llamafactory.model.patcher: High-level patching that uses these utilities.
    llamafactory.model.loader: Model loading that orchestrates these utilities.
"""
