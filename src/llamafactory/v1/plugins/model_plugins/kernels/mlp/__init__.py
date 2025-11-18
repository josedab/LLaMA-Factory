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

"""Provide MLP kernel implementations for v1.

This module contains hardware-optimized kernel implementations for MLP
(Multi-Layer Perceptron) operations, including SwiGLU activation and
Mixture of Experts (MoE) routing. These kernels provide significant
performance improvements on supported hardware.

Key Submodules:
    npu_swiglu: NPU-optimized SwiGLU activation kernel.
    npu_fused_moe: NPU-optimized fused MoE kernel.

See Also:
    llamafactory.v1.plugins.model_plugins.kernels.registry: Kernel registration.
    llamafactory.v1.plugins.model_plugins.kernels.constants: KernelType constants.
"""
