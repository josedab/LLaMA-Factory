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

"""Provide training modules for all LLaMA-Factory training stages.

This package contains training workflows, trainers, callbacks, and utilities
for various training stages including pre-training (PT), supervised fine-tuning
(SFT), reward modeling (RM), reinforcement learning (PPO), direct preference
optimization (DPO), and Kahneman-Tversky optimization (KTO).

Subpackages:
    pt: Pre-training workflow and trainer.
    sft: Supervised fine-tuning workflow, trainer, and metrics.
    rm: Reward modeling workflow, trainer, and metrics.
    ppo: Proximal policy optimization workflow and trainer.
    dpo: Direct preference optimization workflow and trainer.
    kto: Kahneman-Tversky optimization workflow and trainer.
    mca: MCore Adapter workflows for Megatron-Core training.
    ksft: KTransformers SFT workflow for CPU-offloaded training.

Key Modules:
    tuner: Main entry point for training orchestration and model export.
    callbacks: Training callbacks for logging, checkpointing, and integration.
    trainer_utils: Utility functions for optimizers, schedulers, and models.
    test_utils: Testing utilities for model and training validation.
    fp8_utils: FP8 training configuration utilities.

Example:
    >>> from llamafactory.train.tuner import run_exp, export_model
    >>> # Run supervised fine-tuning
    >>> run_exp(args={"stage": "sft", "model_name_or_path": "meta-llama/Llama-2-7b"})
    >>> # Run DPO training
    >>> from llamafactory.train.dpo import run_dpo
    >>> run_dpo(model_args, data_args, training_args, finetuning_args)

See Also:
    - llamafactory.hparams: Argument dataclasses for training configuration.
    - llamafactory.model: Model loading and preparation utilities.
    - llamafactory.data: Dataset loading and processing utilities.
"""
