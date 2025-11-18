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

"""Expose Proximal Policy Optimization (PPO) workflow for RLHF training.

This package provides the PPO workflow for reinforcement learning from human
feedback (RLHF), training models to generate responses that maximize rewards
from a trained reward model.

Key Functions:
    run_ppo: Execute the PPO training workflow with specified arguments.

Example:
    >>> from llamafactory.train.ppo import run_ppo
    >>> run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

See Also:
    - llamafactory.train.ppo.workflow: Full PPO workflow implementation.
    - llamafactory.train.ppo.trainer: Custom PPO trainer extending TRL.
    - llamafactory.train.ppo.ppo_utils: Utility functions for PPO training.
    - llamafactory.train.tuner: Main training orchestration.
"""

from .workflow import run_ppo


__all__ = ["run_ppo"]
