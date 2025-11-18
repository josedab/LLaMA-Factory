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

"""Expose MCore Adapter (MCA) workflows for Megatron-Core based training.

This package provides training workflows using the mcore_adapter library for
high-performance training with Megatron-Core, supporting PT, SFT, and DPO
stages with tensor/pipeline/expert parallelism.

Key Functions:
    run_pt: Execute pre-training workflow with MCore Adapter.
    run_sft: Execute supervised fine-tuning workflow with MCore Adapter.
    run_dpo: Execute DPO training workflow with MCore Adapter.

Example:
    >>> from llamafactory.train.mca import run_sft
    >>> run_sft(model_args, data_args, training_args, finetuning_args, callbacks)

See Also:
    - llamafactory.train.mca.workflow: Full MCA workflow implementations.
    - mcore_adapter: Megatron-Core adapter library.
    - llamafactory.train.tuner: Main training orchestration.
"""

from .workflow import run_dpo, run_pt, run_sft


__all__ = ["run_dpo", "run_pt", "run_sft"]
