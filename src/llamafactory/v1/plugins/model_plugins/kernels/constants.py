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

"""Define enumeration constants for kernel types and device types.

This module provides the enumeration definitions used throughout the kernel
plugin system to identify kernel categories and target hardware devices.

Key Classes:
    KernelType: Enumeration of supported kernel operation types (RMSNORM,
        SWIGLU, FLASH_ATTENTION, ROPE, MOE).
    DeviceType: Enumeration of supported hardware device types (CPU, CUDA,
        NPU, XPU).

Example:
    Use kernel type constants::

        from llamafactory.v1.plugins.model_plugins.kernels.constants import (
            KernelType, DeviceType
        )

        kernel_type = KernelType.RMSNORM
        device_type = DeviceType.NPU

See Also:
    llamafactory.v1.plugins.model_plugins.kernels.registry: Uses these constants.
"""

from enum import Enum


class KernelType(str, Enum):
    RMSNORM = "rmsnorm"
    SWIGLU = "swiglu"
    FLASH_ATTENTION = "flash_attention"
    ROPE = "rope"
    MOE = "moe"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"
    XPU = "xpu"
