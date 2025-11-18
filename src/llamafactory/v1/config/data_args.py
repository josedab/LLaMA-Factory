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

"""Define data-related configuration arguments for v1 training.

This module provides the DataArguments dataclass containing all configuration
parameters related to dataset loading, preprocessing, and data pipeline setup.
These arguments control how training data is loaded, tokenized, and prepared
for model training.

Key Classes:
    DataArguments: Dataclass containing dataset path, directory, cutoff length,
        and other data-related configuration options.

Example:
    Create data arguments with custom settings::

        from llamafactory.v1.config.data_args import DataArguments

        data_args = DataArguments(
            dataset="alpaca_data.json",
            dataset_dir="./data",
            cutoff_len=4096
        )

    Access from parsed configuration::

        from llamafactory.v1.config.parser import get_args

        data_args, *_ = get_args()
        print(f"Dataset: {data_args.dataset}")
        print(f"Cutoff length: {data_args.cutoff_len}")

See Also:
    llamafactory.v1.config.parser: Uses DataArguments in argument parsing.
    llamafactory.v1.core.data_engine: Consumes DataArguments for data loading.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Cutoff length for the dataset."},
    )
