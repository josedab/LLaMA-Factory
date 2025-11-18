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

"""Define OpenAI-compatible data models for API request and response protocols.

This module contains all Pydantic data models that define the structure of
API requests and responses. The models are designed to be compatible with
the OpenAI API specification, allowing clients to use existing OpenAI SDK
code with minimal modifications.

The module defines models for:
    - Chat completion requests and responses (streaming and non-streaming)
    - Function/tool calling definitions and invocations
    - Multimodal input handling (text, images, videos, audio)
    - Model listing and information
    - Score evaluation for reward models

Key Classes:
    ChatCompletionRequest: Input model for chat completion API.
    ChatCompletionResponse: Output model for non-streaming completions.
    ChatCompletionStreamResponse: Output model for streaming completion chunks.
    ScoreEvaluationRequest: Input model for reward model scoring.
    ScoreEvaluationResponse: Output model for score evaluation results.

Key Enums:
    Role: Message roles (user, assistant, system, function, tool).
    Finish: Completion finish reasons (stop, length, tool_calls).

Example:
    Create a chat completion request::

        from llamafactory.api.protocol import (
            ChatCompletionRequest,
            ChatMessage,
            Role
        )

        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[
                ChatMessage(role=Role.SYSTEM, content="You are helpful."),
                ChatMessage(role=Role.USER, content="Hello!")
            ],
            temperature=0.7,
            max_tokens=100,
            stream=False
        )

    Create a multimodal message with image::

        from llamafactory.api.protocol import (
            ChatMessage,
            MultimodalInputItem,
            URL,
            Role
        )

        message = ChatMessage(
            role=Role.USER,
            content=[
                MultimodalInputItem(type="text", text="What's in this image?"),
                MultimodalInputItem(
                    type="image_url",
                    image_url=URL(url="https://example.com/image.png")
                )
            ]
        )

    Define tool/function for the model::

        from llamafactory.api.protocol import FunctionAvailable, FunctionDefinition

        tool = FunctionAvailable(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                }
            )
        )

See Also:
    - :mod:`llamafactory.api.chat`: Uses these models for request processing.
    - :mod:`llamafactory.api.app`: Registers these models with FastAPI endpoints.
    - OpenAI API reference: https://platform.openai.com/docs/api-reference/chat
    - Pydantic documentation: https://docs.pydantic.dev/
"""

import time
from enum import Enum, unique
from typing import Any, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@unique
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["owner"] = "owner"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = []


class Function(BaseModel):
    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: Optional[FunctionDefinition] = None


class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function


class URL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url", "video_url", "audio_url"]
    text: Optional[str] = None
    image_url: Optional[URL] = None
    video_url: Optional[URL] = None
    audio_url: Optional[URL] = None


class ChatMessage(BaseModel):
    role: Role
    content: Optional[Union[str, list[MultimodalInputItem]]] = None
    tool_calls: Optional[list[FunctionCall]] = None


class ChatCompletionMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[list[FunctionCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: Optional[list[FunctionAvailable]] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]


class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: list[str]
    max_length: Optional[int] = None


class ScoreEvaluationResponse(BaseModel):
    id: str
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: list[float]
