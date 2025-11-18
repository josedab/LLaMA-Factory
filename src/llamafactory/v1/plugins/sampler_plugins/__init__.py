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

"""Provide inference sampling backend plugins for v1.

This module contains plugins for various text generation and inference backends.
These plugins enable high-performance text generation using optimized inference
engines like vLLM.

Key Submodules:
    vllm: vLLM inference backend integration for high-throughput generation.

See Also:
    llamafactory.v1.core.chat_sampler: Uses sampler plugins for generation.
    llamafactory.v1.config.sample_args: Sampling configuration parameters.
"""
