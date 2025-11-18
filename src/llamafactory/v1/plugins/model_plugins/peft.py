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

"""Provide parameter-efficient fine-tuning plugins for v1.

This module will contain implementations of parameter-efficient fine-tuning
methods such as LoRA (Low-Rank Adaptation), QLoRA, and other adapter-based
approaches that enable efficient model adaptation with reduced memory footprint.

See Also:
    llamafactory.v1.core.model_engine: Uses PEFT plugins during model setup.
    llamafactory.v1.trainers: Trainers that support PEFT methods.
"""
