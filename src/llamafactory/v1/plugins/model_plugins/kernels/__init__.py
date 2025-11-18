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

"""Provide hardware-optimized kernel plugins for v1.

This module contains the kernel plugin system for applying hardware-specific
optimizations to language models. The system uses a registry-based approach
where kernels register themselves for specific hardware types (NPU, CUDA, etc.)
and are automatically applied based on the runtime environment.

Key Submodules:
    constants: Enumeration definitions for kernel types and device types.
    registry: Global kernel registry and MetaKernel base classes.
    fa: Flash Attention kernel implementations.
    mlp: MLP kernels including SwiGLU and MoE.
    rms_norm: RMSNorm kernel implementations.
    rope: Rotary Position Embedding kernel implementations.

Key Classes:
    KernelRegistry: Singleton registry for kernel implementations.
    MetaKernel: Abstract base class for all kernel plugins.
    KernelType: Enumeration of supported kernel types.
    DeviceType: Enumeration of supported device types.

Key Functions:
    apply_kernel: Apply a MetaKernel to a model.
    discover_kernels: Auto-discover applicable kernels for a model.

Example:
    Register and apply a custom kernel::

        from llamafactory.v1.plugins.model_plugins.kernels.registry import (
            KERNEL_REGISTRY, apply_kernel
        )
        from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import (
            NpuRMSNormKernel
        )

        NpuRMSNormKernel.register_kernel()
        model = apply_kernel(model, NpuRMSNormKernel)

See Also:
    llamafactory.v1.plugins.model_plugins.kernels.registry: Registry implementation.
    llamafactory.v1.plugins.model_plugins.kernels.constants: Type definitions.
"""
