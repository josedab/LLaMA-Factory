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

"""Define model-related configuration arguments for v1 training.

This module provides the ModelArguments dataclass containing all configuration
parameters related to model loading and initialization. These arguments specify
the model path, trust settings for remote code, and other model-specific options.

Key Classes:
    ModelArguments: Dataclass containing model path, trust_remote_code setting,
        and other model loading configuration options.

Example:
    Create model arguments with custom settings::

        from llamafactory.v1.config.model_args import ModelArguments

        model_args = ModelArguments(
            model="meta-llama/Llama-2-7b-hf",
            trust_remote_code=True
        )

    Access from parsed configuration::

        from llamafactory.v1.config.parser import get_args

        _, model_args, *_ = get_args()
        print(f"Model: {model_args.model}")

See Also:
    llamafactory.v1.config.parser: Uses ModelArguments in argument parsing.
    llamafactory.v1.core.model_engine: Consumes ModelArguments for model loading.
"""

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to the model or model identifier from Hugging Face."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code from Hugging Face."},
    )
