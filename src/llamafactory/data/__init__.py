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

r"""
Provide data loading, processing, and collation utilities for LLM fine-tuning.

This module serves as the main entry point for data handling in LlamaFactory.
It provides comprehensive support for loading datasets from various sources,
converting them to standard formats, applying chat templates, tokenizing text,
and collating batches for different training stages including pretraining,
supervised fine-tuning, reward modeling, and reinforcement learning.

Key Classes and Functions:
    get_dataset: Main function to load and preprocess datasets for training.
    get_template_and_fix_tokenizer: Retrieve chat template and fix tokenizer special tokens.
    split_dataset: Split dataset into training and validation sets.
    Template: Chat template class for formatting conversations.
    TEMPLATES: Registry of all available chat templates.
    Role: Enumeration of conversation roles (user, assistant, system, etc.).
    MultiModalDataCollatorForSeq2Seq: Data collator supporting vision-language models.
    SFTDataCollatorWith4DAttentionMask: Data collator with 4D attention mask support.
    PairwiseDataCollatorWithPadding: Data collator for preference learning.
    KTODataCollatorWithPadding: Data collator for KTO training.

Example:
    >>> from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    >>> from llamafactory.hparams import get_train_args
    >>> model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    >>> template = get_template_and_fix_tokenizer(tokenizer, data_args)
    >>> dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", tokenizer=tokenizer)
    >>> train_dataset = dataset_module["train_dataset"]

See Also:
    llamafactory.data.loader: Dataset loading functions.
    llamafactory.data.template: Chat template definitions.
    llamafactory.data.collator: Data collation utilities.
    llamafactory.data.processor: Stage-specific data processors.
"""

from .collator import (
    KTODataCollatorWithPadding,
    MultiModalDataCollatorForSeq2Seq,
    PairwiseDataCollatorWithPadding,
    SFTDataCollatorWith4DAttentionMask,
)
from .data_utils import Role, split_dataset
from .loader import get_dataset
from .template import TEMPLATES, Template, get_template_and_fix_tokenizer


__all__ = [
    "TEMPLATES",
    "KTODataCollatorWithPadding",
    "MultiModalDataCollatorForSeq2Seq",
    "PairwiseDataCollatorWithPadding",
    "Role",
    "SFTDataCollatorWith4DAttentionMask",
    "Template",
    "get_dataset",
    "get_template_and_fix_tokenizer",
    "split_dataset",
]
