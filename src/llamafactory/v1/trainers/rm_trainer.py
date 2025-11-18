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

"""Implement Reward Model trainer for v1.

This module will contain the RMTrainer class for training reward models
that learn to score responses based on human preferences. Reward models
are used in RLHF pipelines to provide feedback signals for policy training.

See Also:
    llamafactory.v1.core.base_trainer: Base trainer class.
    llamafactory.v1.trainers.dpo_trainer: Alternative alignment approach.
"""
