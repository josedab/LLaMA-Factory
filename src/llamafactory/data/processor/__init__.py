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
Provide stage-specific dataset processors for different training objectives.

This package contains dataset processor classes that transform aligned datasets
into model-ready formats for different training stages. Each processor handles
the specific requirements of its training objective, including tokenization,
label masking, and feature construction.

Key Classes:
    DatasetProcessor: Abstract base class for all dataset processors.
    PretrainDatasetProcessor: Processor for language model pretraining.
    SupervisedDatasetProcessor: Processor for supervised fine-tuning (SFT).
    PackedSupervisedDatasetProcessor: SFT processor with sequence packing.
    PairwiseDatasetProcessor: Processor for pairwise preference learning (DPO/ORPO).
    FeedbackDatasetProcessor: Processor for KTO (Kahneman-Tversky Optimization).
    UnsupervisedDatasetProcessor: Processor for unsupervised/PPO training.

Example:
    >>> from llamafactory.data.processor import SupervisedDatasetProcessor
    >>> processor = SupervisedDatasetProcessor(
    ...     template=template,
    ...     tokenizer=tokenizer,
    ...     processor=mm_processor,
    ...     data_args=data_args
    ... )
    >>> model_inputs = processor.preprocess_dataset(examples)

See Also:
    llamafactory.data.loader: Uses processors for dataset preprocessing.
    llamafactory.data.template: Templates used by processors for encoding.
    llamafactory.data.collator: Collators that batch processor outputs.
"""

from .feedback import FeedbackDatasetProcessor
from .pairwise import PairwiseDatasetProcessor
from .pretrain import PretrainDatasetProcessor
from .processor_utils import DatasetProcessor
from .supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from .unsupervised import UnsupervisedDatasetProcessor


__all__ = [
    "DatasetProcessor",
    "FeedbackDatasetProcessor",
    "PackedSupervisedDatasetProcessor",
    "PairwiseDatasetProcessor",
    "PretrainDatasetProcessor",
    "SupervisedDatasetProcessor",
    "UnsupervisedDatasetProcessor",
]
