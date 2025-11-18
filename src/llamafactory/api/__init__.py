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

"""Provide OpenAI-compatible REST API server for LLM inference.

This package implements a FastAPI-based REST API server that exposes
LlamaFactory's chat and scoring capabilities through OpenAI-compatible
endpoints. It supports both streaming and non-streaming chat completions,
multimodal inputs (images, videos, audio), tool/function calling, and
reward model scoring.

The API server can be configured via environment variables for host, port,
API key authentication, and model naming. It includes built-in security
measures against SSRF and LFI vulnerabilities.

Key Modules:
    app: FastAPI application factory and server entry point.
    chat: Chat completion and score evaluation endpoint handlers.
    common: Shared utilities for serialization and security validation.
    protocol: Pydantic data models for OpenAI-compatible request/response formats.

Example:
    Start the API server from command line::

        $ llamafactory-cli api --model_name_or_path meta-llama/Llama-2-7b-chat-hf

    Or programmatically::

        from llamafactory.api.app import run_api
        run_api()

    Make requests to the API::

        import requests
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )

See Also:
    - :mod:`llamafactory.chat`: Core chat model implementation.
    - :mod:`llamafactory.data`: Data processing and role definitions.
    - FastAPI documentation: https://fastapi.tiangolo.com/
    - OpenAI API reference: https://platform.openai.com/docs/api-reference
"""
