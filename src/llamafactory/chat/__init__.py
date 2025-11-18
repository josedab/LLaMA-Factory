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

"""Provide chat model interfaces and inference engines for LLM interaction.

This package provides the core chat functionality for LLaMA-Factory, including
a unified interface for interacting with language models through multiple
inference backends. The package abstracts away the complexity of different
inference engines (HuggingFace, vLLM, SGLang, KTransformers) behind a common
API, enabling seamless switching between backends based on performance needs.

Key Classes:
    BaseEngine: Abstract base class defining the inference engine interface.
    ChatModel: Main entry point for chat-based interactions with models.

Usage Example:
    >>> from llamafactory.chat import ChatModel
    >>>
    >>> # Initialize chat model with configuration
    >>> chat_model = ChatModel({"model_name_or_path": "meta-llama/Llama-2-7b-chat-hf"})
    >>>
    >>> # Single-turn conversation
    >>> messages = [{"role": "user", "content": "Hello, how are you?"}]
    >>> responses = chat_model.chat(messages)
    >>> print(responses[0].response_text)
    >>>
    >>> # Streaming response
    >>> for token in chat_model.stream_chat(messages):
    ...     print(token, end="", flush=True)

See Also:
    llamafactory.hparams: Configuration parameters for model and generation.
    llamafactory.model: Model loading and tokenizer utilities.
    llamafactory.data: Data processing and template management.
"""

from .base_engine import BaseEngine
from .chat_model import ChatModel


__all__ = ["BaseEngine", "ChatModel"]
