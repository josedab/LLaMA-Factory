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

"""Expose Direct Preference Optimization (DPO) workflow for preference alignment.

This package provides the DPO workflow for aligning language models with human
preferences using direct optimization on preference pairs, supporting various
loss types including DPO, IPO, ORPO, SimPO, and BCO.

Key Functions:
    run_dpo: Execute the DPO training workflow with specified arguments.

Example:
    >>> from llamafactory.train.dpo import run_dpo
    >>> run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)

See Also:
    - llamafactory.train.dpo.workflow: Full DPO workflow implementation.
    - llamafactory.train.dpo.trainer: Custom DPO trainer extending TRL.
    - llamafactory.train.tuner: Main training orchestration.
"""

from .workflow import run_dpo


__all__ = ["run_dpo"]
