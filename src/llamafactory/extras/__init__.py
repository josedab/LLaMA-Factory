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

"""Provide extra utilities and helpers for LLaMA-Factory.

This package contains various utility modules that support the core
functionality of LLaMA-Factory, including logging, environment detection,
constants, package management, plotting utilities, and miscellaneous
helper functions.

Modules:
    constants: Global constants, enums, and model configurations.
    env: Environment information and version detection.
    logging: Custom logging system with rank-aware capabilities.
    misc: Miscellaneous utility functions for device, memory, and version handling.
    packages: Package availability checking utilities.
    ploting: Training metrics visualization and plotting functions.

Example:
    Access utilities from the extras package::

        from llamafactory.extras import logging
        from llamafactory.extras.constants import SUPPORTED_MODELS
        from llamafactory.extras.misc import get_current_device

        logger = logging.get_logger(__name__)
        device = get_current_device()

See Also:
    llamafactory.hparams: Hyperparameter configurations.
    llamafactory.data: Data loading and processing utilities.
    llamafactory.model: Model loading and configuration.
"""
