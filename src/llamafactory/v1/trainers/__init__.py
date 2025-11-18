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

"""Provide specialized trainer implementations for v1.

This module contains trainer classes for different training paradigms in the
v1 system. Each trainer extends the BaseTrainer and implements task-specific
loss computation and training procedures.

Key Submodules:
    sft_trainer: Supervised Fine-Tuning trainer for instruction following.
    dpo_trainer: Direct Preference Optimization trainer for alignment.
    rm_trainer: Reward Model trainer for preference learning.

Key Classes:
    SFTTrainer: Trainer for supervised fine-tuning on instruction-response data.
    DPOTrainer: Trainer for direct preference optimization with chosen/rejected pairs.
    RMTrainer: Trainer for reward model training.

Key Functions:
    run_sft: Entry point function to run SFT training.

Example:
    Run SFT training::

        from llamafactory.v1.trainers.sft_trainer import run_sft
        run_sft()

    Use SFTTrainer directly::

        from llamafactory.v1.trainers.sft_trainer import SFTTrainer

        trainer = SFTTrainer(training_args, model, processor, data_loader)
        trainer.fit()

See Also:
    llamafactory.v1.core.base_trainer: Base trainer class.
    llamafactory.v1.config.training_args: Training configuration.
"""
