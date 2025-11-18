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

"""Data-related constants.

This module contains file type mappings, data format constants,
and evaluation-related constants.
"""


# File extension to data type mapping
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

# Special token index for ignored labels in loss computation
IGNORE_INDEX = -100

# Multiple choice answer options
CHOICES = ["A", "B", "C", "D"]

# Evaluation subjects for benchmarks (e.g., MMLU)
SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]


__all__ = [
    "FILEEXT2TYPE",
    "IGNORE_INDEX",
    "CHOICES",
    "SUBJECTS",
]
