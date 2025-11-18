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

"""Provide configuration argument dataclasses for v1 training.

This module contains the configuration system for LLaMA Factory v1, organizing
all training parameters into logical dataclass groups. The configuration supports
multiple input formats including YAML files, JSON files, and command-line
arguments, with automatic parsing and validation.

The configuration is divided into four main argument groups:
    - DataArguments: Dataset paths, directories, and preprocessing options
    - ModelArguments: Model paths, trust settings, and loading options
    - TrainingArguments: Training hyperparameters and output settings
    - SampleArguments: Generation and sampling parameters

Key Classes:
    DataArguments: Configuration for dataset loading and preprocessing.
    ModelArguments: Configuration for model loading and initialization.
    TrainingArguments: Configuration for training loop and optimization.
    SampleArguments: Configuration for text generation sampling.

Key Functions:
    get_args: Parse and return all argument groups from various input sources.

Example:
    Parse arguments from a YAML config file::

        from llamafactory.v1.config import get_args

        data_args, model_args, training_args, sample_args = get_args()

    Parse with custom dictionary::

        args = {
            "model": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca",
            "output_dir": "./output"
        }
        data_args, model_args, training_args, sample_args = get_args(args)

See Also:
    llamafactory.v1.config.parser: Argument parsing implementation.
    llamafactory.v1.config.data_args: Data argument definitions.
    llamafactory.v1.config.model_args: Model argument definitions.
    llamafactory.v1.config.training_args: Training argument definitions.
"""
