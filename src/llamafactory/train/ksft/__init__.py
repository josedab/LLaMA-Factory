# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

"""Expose KTransformers SFT workflow for CPU-offloaded training.

This package provides the supervised fine-tuning workflow using KTransformers
for efficient training with CPU offloading and LoRA optimization on consumer
hardware with limited GPU memory.

Key Functions:
    run_sft: Execute SFT workflow with KTransformers trainer.

Example:
    >>> from llamafactory.train.ksft import run_sft
    >>> run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

See Also:
    - llamafactory.train.ksft.workflow: Full KTransformers SFT workflow.
    - ktransformers.sft.lora.KTrainer: KTransformers trainer for SFT.
    - llamafactory.train.tuner: Main training orchestration.
"""

from .workflow import run_sft


__all__ = ["run_sft"]
