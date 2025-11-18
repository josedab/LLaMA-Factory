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

"""Type definitions for hyperparameters module."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from .data_args import DataArguments
    from .evaluation_args import EvaluationArguments
    from .finetuning_args import FinetuningArguments
    from .generating_args import GeneratingArguments
    from .model_args import ModelArguments
    from .training_args import RayArguments, TrainingArguments


# Type variables for argument classes
ModelArgsType = TypeVar("ModelArgsType", bound="ModelArguments")
DataArgsType = TypeVar("DataArgsType", bound="DataArguments")
TrainingArgsType = TypeVar("TrainingArgsType", bound="TrainingArguments")
FinetuningArgsType = TypeVar("FinetuningArgsType", bound="FinetuningArguments")
GeneratingArgsType = TypeVar("GeneratingArgsType", bound="GeneratingArguments")
EvaluationArgsType = TypeVar("EvaluationArgsType", bound="EvaluationArguments")
RayArgsType = TypeVar("RayArgsType", bound="RayArguments")

# Type aliases for common argument tuples
TrainArgsTuple = tuple[
    "ModelArguments",
    "DataArguments",
    "TrainingArguments",
    "FinetuningArguments",
    "GeneratingArguments",
]

InferArgsTuple = tuple[
    "ModelArguments",
    "DataArguments",
    "FinetuningArguments",
    "GeneratingArguments",
]

EvalArgsTuple = tuple[
    "ModelArguments",
    "DataArguments",
    "EvaluationArguments",
    "FinetuningArguments",
]

__all__ = [
    "DataArgsType",
    "EvalArgsTuple",
    "EvaluationArgsType",
    "FinetuningArgsType",
    "GeneratingArgsType",
    "InferArgsTuple",
    "ModelArgsType",
    "RayArgsType",
    "TrainArgsTuple",
    "TrainingArgsType",
]
