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

"""Define the abstract base class for inference engines.

This module provides the foundational interface that all inference engine
implementations must follow. It defines the common contract for chat-based
text generation, streaming responses, and reward model scoring. Each engine
implementation (HuggingFace, vLLM, SGLang, KTransformers) inherits from
BaseEngine and provides backend-specific optimizations while maintaining
API compatibility.

The module also defines the Response dataclass for standardized output format
across all engines, containing response text, token counts, and finish reasons.

Key Classes:
    Response: Dataclass containing generation output and metadata.
    BaseEngine: Abstract base class for all inference engine implementations.

Key Methods (BaseEngine):
    chat: Generate complete responses for given messages.
    stream_chat: Stream response tokens incrementally.
    get_scores: Compute reward model scores for input texts.

Usage Example:
    >>> from llamafactory.chat.base_engine import BaseEngine, Response
    >>>
    >>> # Custom engine implementation
    >>> class CustomEngine(BaseEngine):
    ...     async def chat(self, messages, system=None, tools=None,
    ...                    images=None, videos=None, audios=None, **kwargs):
    ...         # Implementation here
    ...         return [Response(
    ...             response_text="Hello!",
    ...             response_length=1,
    ...             prompt_length=10,
    ...             finish_reason="stop"
    ...         )]
    ...
    ...     async def stream_chat(self, messages, **kwargs):
    ...         yield "Hello!"
    ...
    ...     async def get_scores(self, batch_input, **kwargs):
    ...         return [0.5] * len(batch_input)

See Also:
    llamafactory.chat.hf_engine: HuggingFace Transformers implementation.
    llamafactory.chat.vllm_engine: vLLM high-throughput implementation.
    llamafactory.chat.sglang_engine: SGLang optimized implementation.
    llamafactory.chat.kt_engine: KTransformers implementation.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from vllm import AsyncLLMEngine

    from ..data import Template
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from ..extras.constants import EngineName
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class BaseEngine(ABC):
    r"""Base class for inference engine of chat models.

    Must implements async methods: chat(), stream_chat() and get_scores().
    """

    name: "EngineName"
    model: Union["PreTrainedModel", "AsyncLLMEngine"]
    tokenizer: "PreTrainedTokenizer"
    can_generate: bool
    template: "Template"
    generating_args: dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        r"""Initialize an inference engine."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Get a list of responses of the chat model."""
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""Get the response token-by-token of the chat model."""
        ...

    @abstractmethod
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Get a list of scores of the reward model."""
        ...
