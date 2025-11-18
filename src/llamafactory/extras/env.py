# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/commands/env.py
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

"""Provide environment information and version detection for LLaMA-Factory.

This module contains version information and utilities for displaying
the current environment configuration, including Python version, PyTorch
version, hardware information, and installed package versions. It is
primarily used for debugging and issue reporting.

Constants:
    VERSION: Current version string of LLaMA-Factory.

Functions:
    print_env: Display comprehensive environment information.

Example:
    Print environment information for debugging::

        from llamafactory.extras.env import VERSION, print_env

        # Check current version
        print(f"LLaMA-Factory version: {VERSION}")

        # Print full environment details
        print_env()
        # Output includes:
        # - LLaMA-Factory version
        # - Platform and Python version
        # - PyTorch, Transformers, Datasets versions
        # - GPU/NPU type, memory, and count
        # - Optional packages (DeepSpeed, vLLM, etc.)

See Also:
    llamafactory.extras.packages: Package availability checking.
    llamafactory.extras.misc: Device and memory utilities.
"""


from collections import OrderedDict


VERSION = "0.9.4.dev0"


def print_env() -> None:
    import os
    import platform

    import accelerate
    import datasets
    import peft
    import torch
    import transformers
    from transformers.utils import is_torch_cuda_available, is_torch_npu_available

    info = OrderedDict(
        {
            "`llamafactory` version": VERSION,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyTorch version": torch.__version__,
            "Transformers version": transformers.__version__,
            "Datasets version": datasets.__version__,
            "Accelerate version": accelerate.__version__,
            "PEFT version": peft.__version__,
        }
    )

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()
        info["GPU number"] = torch.cuda.device_count()
        info["GPU memory"] = f"{torch.cuda.mem_get_info()[1] / (1024**3):.2f}GB"

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    try:
        import trl  # type: ignore

        info["TRL version"] = trl.__version__
    except Exception:
        pass

    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes  # type: ignore

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info["vLLM version"] = vllm.__version__
    except Exception:
        pass

    try:
        import subprocess

        commit_info = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_info.stdout.strip()
        info["Git commit"] = commit_hash
    except Exception:
        pass

    if os.path.exists("data"):
        info["Default data directory"] = "detected"
    else:
        info["Default data directory"] = "not detected"

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")
