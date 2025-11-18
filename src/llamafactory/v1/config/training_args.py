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

"""Define training-related configuration arguments for v1 training.

This module provides the TrainingArguments dataclass containing all configuration
parameters related to the training loop and optimization. These arguments control
batch sizes, learning rates, precision settings, and output directories.

Key Classes:
    TrainingArguments: Dataclass containing output_dir, batch sizes, learning_rate,
        precision (bf16), and other training configuration options.

Example:
    Create training arguments with custom settings::

        from llamafactory.v1.config.training_args import TrainingArguments

        training_args = TrainingArguments(
            output_dir="./output",
            micro_batch_size=4,
            global_batch_size=32,
            learning_rate=2e-5,
            bf16=True
        )

    Access from parsed configuration::

        from llamafactory.v1.config.parser import get_args

        _, _, training_args, _ = get_args()
        print(f"Learning rate: {training_args.learning_rate}")

See Also:
    llamafactory.v1.config.parser: Uses TrainingArguments in argument parsing.
    llamafactory.v1.core.base_trainer: Consumes TrainingArguments for training loop.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="",
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int = field(
        default=1,
        metadata={"help": "Global batch size for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16 for training."},
    )
