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

"""Provide hyperparameter argument classes and parsing utilities for LLaMA-Factory.

This package contains dataclass definitions for configuring all aspects of model
training, inference, and evaluation in LLaMA-Factory. It includes argument classes
for data processing, model configuration, fine-tuning methods, generation settings,
and training parameters, along with utility functions for parsing these arguments
from configuration files or command-line inputs.

Key Classes:
    DataArguments: Configuration for dataset loading and preprocessing.
    EvaluationArguments: Parameters for model evaluation tasks.
    FinetuningArguments: Settings for various fine-tuning methods (LoRA, freeze, etc.).
    GeneratingArguments: Decoding and generation parameters.
    ModelArguments: Model architecture and loading configurations.
    RayArguments: Distributed training settings for Ray.
    TrainingArguments: General training hyperparameters.

Key Functions:
    get_train_args: Parse and validate arguments for training.
    get_infer_args: Parse arguments for inference.
    get_eval_args: Parse arguments for evaluation.
    get_ray_args: Parse Ray-specific training arguments.
    read_args: Read arguments from config files or command line.

Example:
    >>> from llamafactory.hparams import get_train_args
    >>> model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    >>> from llamafactory.hparams import ModelArguments, DataArguments
    >>> # Use dataclasses directly for custom configuration

See Also:
    transformers.TrainingArguments: Base class for training arguments.
    transformers.HfArgumentParser: Parser used for argument handling.
"""

from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .parser import get_eval_args, get_infer_args, get_ray_args, get_train_args, read_args
from .training_args import RayArguments, TrainingArguments


__all__ = [
    "DataArguments",
    "EvaluationArguments",
    "FinetuningArguments",
    "GeneratingArguments",
    "ModelArguments",
    "RayArguments",
    "TrainingArguments",
    "get_eval_args",
    "get_infer_args",
    "get_ray_args",
    "get_train_args",
    "read_args",
]
