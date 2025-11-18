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

"""Provide training backend plugins for v1.

This module contains plugins for various training backends and distributed
training strategies. These plugins enable training across different hardware
configurations and distributed setups.

Key Submodules:
    distributed: Distributed training plugins including Accelerate integration
        and device detection utilities.

See Also:
    llamafactory.v1.core.base_trainer: Uses trainer plugins for distributed setup.
    llamafactory.v1.trainers: Trainer implementations that consume these plugins.
"""
