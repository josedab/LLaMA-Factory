# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/import_utils.py
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

"""Provide package availability checking utilities for LLaMA-Factory.

This module contains functions to check whether optional packages are
installed and to verify package versions. It enables graceful degradation
when optional dependencies are not available and allows for conditional
feature enabling based on installed packages.

Functions:
    is_pyav_available: Check if PyAV (av) is installed.
    is_librosa_available: Check if librosa is installed.
    is_fastapi_available: Check if FastAPI is installed.
    is_galore_available: Check if GaLore optimizer is installed.
    is_apollo_available: Check if Apollo optimizer is installed.
    is_jieba_available: Check if jieba tokenizer is installed.
    is_gradio_available: Check if Gradio is installed.
    is_matplotlib_available: Check if matplotlib is installed.
    is_mcore_adapter_available: Check if mcore_adapter is installed.
    is_pillow_available: Check if Pillow (PIL) is installed.
    is_ray_available: Check if Ray is installed.
    is_kt_available: Check if KTransformers is installed.
    is_requests_available: Check if requests is installed.
    is_rouge_available: Check if rouge_chinese is installed.
    is_safetensors_available: Check if safetensors is installed.
    is_sglang_available: Check if SGLang is installed.
    is_starlette_available: Check if sse_starlette is installed.
    is_transformers_version_greater_than: Check transformers version.
    is_torch_version_greater_than: Check PyTorch version.
    is_uvicorn_available: Check if uvicorn is installed.
    is_vllm_available: Check if vLLM is installed.

Example:
    Check package availability before using optional features::

        from llamafactory.extras.packages import (
            is_gradio_available,
            is_vllm_available,
            is_transformers_version_greater_than,
        )

        # Conditional import based on availability
        if is_gradio_available():
            import gradio as gr
            # Launch web UI
        else:
            print("Gradio not installed, web UI unavailable")

        # Check version requirements
        if is_transformers_version_greater_than("4.40.0"):
            # Use newer API features
            pass

        # Check inference engine availability
        if is_vllm_available():
            from vllm import LLM
        else:
            # Fall back to standard inference
            pass

See Also:
    llamafactory.extras.misc: Version checking with error handling.
    llamafactory.extras.env: Environment information display.
"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_pyav_available():
    return _is_package_available("av")


def is_librosa_available():
    return _is_package_available("librosa")


def is_fastapi_available():
    return _is_package_available("fastapi")


def is_galore_available():
    return _is_package_available("galore_torch")


def is_apollo_available():
    return _is_package_available("apollo_torch")


def is_jieba_available():
    return _is_package_available("jieba")


def is_gradio_available():
    return _is_package_available("gradio")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_mcore_adapter_available():
    return _is_package_available("mcore_adapter")


def is_pillow_available():
    return _is_package_available("PIL")


def is_ray_available():
    return _is_package_available("ray")


def is_kt_available():
    return _is_package_available("ktransformers")


def is_requests_available():
    return _is_package_available("requests")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_safetensors_available():
    return _is_package_available("safetensors")


def is_sglang_available():
    return _is_package_available("sglang")


def is_starlette_available():
    return _is_package_available("sse_starlette")


@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str):
    return _get_package_version("torch") >= version.parse(content)


def is_uvicorn_available():
    return _is_package_available("uvicorn")


def is_vllm_available():
    return _is_package_available("vllm")
