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

"""Implement NPU-optimized RMSNorm kernel.

This module provides the NpuRMSNormKernel class which replaces standard RMSNorm
forward methods with NPU-optimized implementations using torch_npu.npu_rms_norm.
The kernel automatically detects RMSNorm modules in the model and replaces their
forward methods.

Key Classes:
    NpuRMSNormKernel: MetaRMSNormKernel implementation that applies NPU-optimized
        normalization to matching RMSNorm modules.

Key Functions:
    _npu_rms_forward: NPU-optimized forward function for RMSNorm modules that
        maintains numerical consistency with baseline behavior.

Example:
    Apply RMSNorm kernel to a model::

        from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import (
            NpuRMSNormKernel
        )
        from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel

        model = apply_kernel(model, NpuRMSNormKernel)

See Also:
    llamafactory.v1.plugins.model_plugins.kernels.registry: MetaRMSNormKernel base.
    llamafactory.v1.plugins.model_plugins.kernels.constants: KernelType.RMSNORM.
"""

import re
import types

from .....extras.types import HFModel
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..constants import DeviceType, KernelType
from ..registry import KERNEL_REGISTRY, MetaRMSNormKernel


def _npu_rms_forward(self, hidden_states):
    """NPU forward implementation for RMSNorm.

    Args:
        self: RMSNorm module instance with `weight` and `variance_epsilon`.
        hidden_states: Input hidden states tensor, same shape as the baseline.

    Returns:
        Normalized tensor consistent with the baseline RMSNorm behavior.
    """
    import torch_npu

    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


class NpuRMSNormKernel(MetaRMSNormKernel):
    """NPU kernel wrapper for RMSNorm that applies the replacement within a model."""

    device = DeviceType.NPU
    kernel = _npu_rms_forward

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.RMSNORM, device_type=DeviceType.NPU):
        """Register the NPU RMSNorm forward implementation to the global registry."""
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> HFModel:
        """Iterate the model and apply NPU-optimized forward to matched RMSNorm modules.

        Key points:
        - Match modules whose class name contains "RMSNorm" (case-insensitive).
        - Bind `_npu_rms_forward` as an instance method via `types.MethodType` to
          replace the original `forward`.
        - Do not modify weights, hyperparameters, or module structure to ensure
          numerical behavior and interface consistency.
        """
        if not is_torch_npu_available():
            return model

        rms_norm_pattern = re.compile("RMSNorm", re.IGNORECASE)

        for name, module in model.named_modules():
            # Match any module whose class name contains "RMSNorm"
            if re.search(rms_norm_pattern, module.__class__.__name__):
                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                module.forward = types.MethodType(cls.kernel, module)

        return model
