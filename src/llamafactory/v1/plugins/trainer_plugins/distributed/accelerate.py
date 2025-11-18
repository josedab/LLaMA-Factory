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

"""Provide accelerator detection utilities for v1.

This module provides utility functions for detecting the available hardware
accelerator in the current environment. It supports CUDA, NPU, XPU, and MPS
devices, with cached results for efficient repeated queries.

Key Functions:
    get_available_accelerator: Get the current hardware accelerator device.
    is_torch_npu_available: Check if NPU is available (cached).
    is_torch_cuda_available: Check if CUDA is available (cached).
    is_torch_xpu_available: Check if XPU is available (cached).
    is_torch_mps_available: Check if MPS is available (cached).

Example:
    Check available accelerator::

        from llamafactory.v1.plugins.trainer_plugins.distributed.accelerate import (
            get_available_accelerator,
            is_torch_npu_available
        )

        device = get_available_accelerator()
        print(f"Using device: {device}")

        if is_torch_npu_available():
            print("NPU is available")

Note:
    The get_available_accelerator function requires torch>=2.7.0. Earlier versions
    will raise an AttributeError or RuntimeError.

See Also:
    llamafactory.v1.plugins.model_plugins.kernels: Uses device detection for kernel selection.
"""

from functools import lru_cache

import torch


def get_available_accelerator():
    """Get available accelerator in current environment.

    Note: this api requires torch>=2.7.0, 2.6 or lower will get an AttributeError or RuntimeError
    """
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        return torch.device("cpu")
    return accelerator


@lru_cache
def is_torch_npu_available():
    return get_available_accelerator().type == "npu"


@lru_cache
def is_torch_cuda_available():
    return get_available_accelerator().type == "cuda"


@lru_cache
def is_torch_xpu_available():
    return get_available_accelerator().type == "xpu"


@lru_cache
def is_torch_mps_available():
    return get_available_accelerator().type == "mps"
