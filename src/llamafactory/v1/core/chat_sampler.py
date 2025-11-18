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

"""Implement text generation sampler for v1 inference.

This module provides the ChatSampler class for generating text responses from
language models during inference. It handles sampling parameters and generation
configuration to produce high-quality text outputs.

The ChatSampler abstracts away the complexity of text generation, providing a
clean interface for chat-based interactions with the model while supporting
various sampling strategies and parameters.

Key Classes:
    ChatSampler: Main sampler class that manages generation parameters and
        produces text outputs from model inputs.

Example:
    Create and use a chat sampler::

        from llamafactory.v1.config.sample_args import SampleArguments
        from llamafactory.v1.core.chat_sampler import ChatSampler

        sample_args = SampleArguments(max_new_tokens=256)
        sampler = ChatSampler(sample_args)
        # Use sampler for generation

See Also:
    llamafactory.v1.config.sample_args: Sampling configuration parameters.
    llamafactory.v1.plugins.sampler_plugins: Sampler plugin implementations.
"""

from ..config.sample_args import SampleArguments


class ChatSampler:
    def __init__(self, sample_args: SampleArguments) -> None:
        self.args = sample_args
