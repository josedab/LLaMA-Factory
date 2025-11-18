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

"""Default values, paths, and configuration constants.

This module contains default values, checkpoint names, log file names,
and other configuration-related constants.
"""

import os

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


# Checkpoint and weights names
CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"

# Log file names
RUNNING_LOG = "running_log.txt"

TRAINER_LOG = "trainer_log.jsonl"

# Configuration file names
DATA_CONFIG = "dataset_info.json"

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

SWANLAB_CONFIG = "swanlab_public_config.json"

TRAINING_ARGS = "training_args.yaml"

# Layer normalization names
LAYERNORM_NAMES = {"norm", "ln"}

# Placeholder tokens for multimodal content
AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")

IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")


__all__ = [
    "CHECKPOINT_NAMES",
    "V_HEAD_WEIGHTS_NAME",
    "V_HEAD_SAFE_WEIGHTS_NAME",
    "RUNNING_LOG",
    "TRAINER_LOG",
    "DATA_CONFIG",
    "LLAMABOARD_CONFIG",
    "SWANLAB_CONFIG",
    "TRAINING_ARGS",
    "LAYERNORM_NAMES",
    "AUDIO_PLACEHOLDER",
    "IMAGE_PLACEHOLDER",
    "VIDEO_PLACEHOLDER",
    # Re-export from dependencies
    "SAFE_ADAPTER_WEIGHTS_NAME",
    "ADAPTER_WEIGHTS_NAME",
    "SAFE_WEIGHTS_INDEX_NAME",
    "SAFE_WEIGHTS_NAME",
    "WEIGHTS_INDEX_NAME",
    "WEIGHTS_NAME",
]
