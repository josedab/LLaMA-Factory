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

from __future__ import annotations

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> Version:
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_pyav_available() -> bool:
    return _is_package_available("av")


def is_librosa_available() -> bool:
    return _is_package_available("librosa")


def is_fastapi_available() -> bool:
    return _is_package_available("fastapi")


def is_galore_available() -> bool:
    return _is_package_available("galore_torch")


def is_apollo_available() -> bool:
    return _is_package_available("apollo_torch")


def is_jieba_available() -> bool:
    return _is_package_available("jieba")


def is_gradio_available() -> bool:
    return _is_package_available("gradio")


def is_matplotlib_available() -> bool:
    return _is_package_available("matplotlib")


def is_mcore_adapter_available() -> bool:
    return _is_package_available("mcore_adapter")


def is_pillow_available() -> bool:
    return _is_package_available("PIL")


def is_ray_available() -> bool:
    return _is_package_available("ray")


def is_kt_available() -> bool:
    return _is_package_available("ktransformers")


def is_requests_available() -> bool:
    return _is_package_available("requests")


def is_rouge_available() -> bool:
    return _is_package_available("rouge_chinese")


def is_safetensors_available() -> bool:
    return _is_package_available("safetensors")


def is_sglang_available() -> bool:
    return _is_package_available("sglang")


def is_starlette_available() -> bool:
    return _is_package_available("sse_starlette")


@lru_cache
def is_transformers_version_greater_than(content: str) -> bool:
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str) -> bool:
    return _get_package_version("torch") >= version.parse(content)


def is_uvicorn_available() -> bool:
    return _is_package_available("uvicorn")


def is_vllm_available() -> bool:
    return _is_package_available("vllm")
