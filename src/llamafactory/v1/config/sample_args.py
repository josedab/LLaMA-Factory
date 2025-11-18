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

"""Define sampling-related configuration arguments for v1 inference.

This module provides the SampleArguments dataclass containing all configuration
parameters related to text generation and sampling. These arguments control
how the model generates text during inference, including maximum token limits
and other generation parameters.

Key Classes:
    SampleArguments: Dataclass containing max_new_tokens and other generation
        configuration options for text sampling.

Example:
    Create sample arguments with custom settings::

        from llamafactory.v1.config.sample_args import SampleArguments

        sample_args = SampleArguments(
            max_new_tokens=256
        )

    Access from parsed configuration::

        from llamafactory.v1.config.parser import get_args

        *_, sample_args = get_args()
        print(f"Max new tokens: {sample_args.max_new_tokens}")

See Also:
    llamafactory.v1.config.parser: Uses SampleArguments in argument parsing.
    llamafactory.v1.core.chat_sampler: Consumes SampleArguments for generation.
"""

from dataclasses import dataclass, field


@dataclass
class SampleArguments:
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
