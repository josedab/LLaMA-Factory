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

"""Protocol definitions for inference engines and chat components."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable


if TYPE_CHECKING:
    from .base_engine import Response


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for inference engines supporting chat functionality.

    This protocol defines the interface that all inference engines must implement
    for duck typing support. Use this for type hints when you need to accept
    any inference engine without requiring inheritance from BaseEngine.
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
        audios: Optional[list[Any]] = None,
        **input_kwargs: Any,
    ) -> list["Response"]:
        """Get a list of responses from the chat model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            system: Optional system prompt.
            tools: Optional tools specification.
            images: Optional list of image inputs.
            videos: Optional list of video inputs.
            audios: Optional list of audio inputs.
            **input_kwargs: Additional keyword arguments.

        Returns:
            List of Response objects.
        """
        ...

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
        audios: Optional[list[Any]] = None,
        **input_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Get the response token-by-token from the chat model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            system: Optional system prompt.
            tools: Optional tools specification.
            images: Optional list of image inputs.
            videos: Optional list of video inputs.
            audios: Optional list of audio inputs.
            **input_kwargs: Additional keyword arguments.

        Yields:
            Response text token by token.
        """
        ...

    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs: Any,
    ) -> list[float]:
        """Get scores from a reward model.

        Args:
            batch_input: List of input strings to score.
            **input_kwargs: Additional keyword arguments.

        Returns:
            List of float scores.
        """
        ...


@runtime_checkable
class Tokenizable(Protocol):
    """Protocol for objects that can tokenize text."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        """Decode token IDs to text."""
        ...


@runtime_checkable
class Generatable(Protocol):
    """Protocol for models that can generate text."""

    def generate(
        self,
        input_ids: Any,
        **kwargs: Any,
    ) -> Any:
        """Generate text from input IDs."""
        ...


__all__ = [
    "Generatable",
    "InferenceEngine",
    "Tokenizable",
]
