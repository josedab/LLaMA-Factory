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

"""Provide vLLM inference backend integration for v1.

This module will contain the vLLM sampler plugin for high-throughput text
generation using the vLLM inference engine. vLLM provides optimized inference
with continuous batching and PagedAttention for efficient memory usage.

See Also:
    llamafactory.v1.plugins.sampler_plugins: Sampler plugin system.
    llamafactory.v1.core.chat_sampler: Uses vLLM backend for generation.
"""
