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

"""Implement base trainer and data collator for v1 training loop.

This module provides the foundational training infrastructure for LLaMA Factory v1,
including the BaseTrainer class that implements the common training loop logic and
the DataCollator class for batch preparation. These components serve as building
blocks for specialized trainers like SFTTrainer and DPOTrainer.

The BaseTrainer implements a flexible training loop that supports custom optimizers,
learning rate schedulers, and data loaders. Specialized trainers extend this base
class to implement task-specific loss functions and training procedures.

Key Classes:
    DataCollator: Collates individual feature dictionaries into batched tensors
        for efficient GPU processing.
    BaseTrainer: Base trainer class providing the core training loop, optimizer
        setup, and checkpointing functionality.

Example:
    Create a custom trainer by extending BaseTrainer::

        from llamafactory.v1.core.base_trainer import BaseTrainer

        class CustomTrainer(BaseTrainer):
            def compute_loss(self, batch):
                outputs = self.model(**batch)
                return outputs.loss

        trainer = CustomTrainer(args, model, processor, dataset, data_collator)
        trainer.fit()

    Use the DataCollator directly::

        from llamafactory.v1.core.base_trainer import DataCollator

        collator = DataCollator(processor)
        batch = collator(features)

See Also:
    llamafactory.v1.trainers.sft_trainer: SFT trainer implementation.
    llamafactory.v1.trainers.dpo_trainer: DPO trainer implementation.
    llamafactory.v1.config.training_args: Training configuration.
"""

from typing import Any

from ..config.training_args import TrainingArguments
from ..extras.types import Model, Processor, Tensor, TorchDataset


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Collate features into a batch."""
        for feature in features:
            pass

        # sft: messages
        # dpo: chosen_messages, rejected_messages


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: Model,
        processor: Processor,
        dataset: TorchDataset,
        data_collator: DataCollator,
    ) -> None:
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.data_collator = data_collator
        self.optimizer = None
        self.lr_scheduler = None

    def create_dataloader(self) -> None:
        pass

    def fit(self) -> None:
        pass
