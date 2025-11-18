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

"""Provide NPU-optimized fused Mixture of Experts kernel.

This module will contain the NPU-optimized implementation of fused Mixture
of Experts (MoE) kernel for efficient expert routing and computation on
Ascend NPU hardware.

See Also:
    llamafactory.v1.plugins.model_plugins.kernels.registry: MetaMoEKernel base class.
    llamafactory.v1.plugins.model_plugins.kernels.constants: KernelType.MOE.
"""
