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

"""Constants module - backwards compatible imports.

Import from submodules for new code:
    from llamafactory.extras.constants.models import SUPPORTED_MODELS

Or use compatibility imports:
    from llamafactory.extras.constants import SUPPORTED_MODELS
"""

# Data-related constants
from .data import (
    CHOICES,
    FILEEXT2TYPE,
    IGNORE_INDEX,
    SUBJECTS,
)

# Default values and paths
from .defaults import (
    ADAPTER_WEIGHTS_NAME,
    AUDIO_PLACEHOLDER,
    CHECKPOINT_NAMES,
    DATA_CONFIG,
    IMAGE_PLACEHOLDER,
    LAYERNORM_NAMES,
    LLAMABOARD_CONFIG,
    RUNNING_LOG,
    SAFE_ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SWANLAB_CONFIG,
    TRAINER_LOG,
    TRAINING_ARGS,
    V_HEAD_SAFE_WEIGHTS_NAME,
    V_HEAD_WEIGHTS_NAME,
    VIDEO_PLACEHOLDER,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

# Model-related constants
from .models import (
    DEFAULT_TEMPLATE,
    DownloadSource,
    MCA_SUPPORTED_MODELS,
    MOD_SUPPORTED_MODELS,
    MULTIMODAL_SUPPORTED_MODELS,
    SUPPORTED_CLASS_FOR_S2ATTN,
    SUPPORTED_MODELS,
    register_model_group,
)

# Training-related constants
from .training import (
    METHODS,
    PEFT_METHODS,
    STAGES_USE_PAIR_DATA,
    TRAINING_STAGES,
    AttentionFunction,
    EngineName,
    QuantizationMethod,
    RopeScaling,
)


__all__ = [
    # Data
    "CHOICES",
    "FILEEXT2TYPE",
    "IGNORE_INDEX",
    "SUBJECTS",
    # Defaults
    "ADAPTER_WEIGHTS_NAME",
    "AUDIO_PLACEHOLDER",
    "CHECKPOINT_NAMES",
    "DATA_CONFIG",
    "IMAGE_PLACEHOLDER",
    "LAYERNORM_NAMES",
    "LLAMABOARD_CONFIG",
    "RUNNING_LOG",
    "SAFE_ADAPTER_WEIGHTS_NAME",
    "SAFE_WEIGHTS_INDEX_NAME",
    "SAFE_WEIGHTS_NAME",
    "SWANLAB_CONFIG",
    "TRAINER_LOG",
    "TRAINING_ARGS",
    "V_HEAD_SAFE_WEIGHTS_NAME",
    "V_HEAD_WEIGHTS_NAME",
    "VIDEO_PLACEHOLDER",
    "WEIGHTS_INDEX_NAME",
    "WEIGHTS_NAME",
    # Models
    "DEFAULT_TEMPLATE",
    "DownloadSource",
    "MCA_SUPPORTED_MODELS",
    "MOD_SUPPORTED_MODELS",
    "MULTIMODAL_SUPPORTED_MODELS",
    "SUPPORTED_CLASS_FOR_S2ATTN",
    "SUPPORTED_MODELS",
    "register_model_group",
    # Training
    "METHODS",
    "PEFT_METHODS",
    "STAGES_USE_PAIR_DATA",
    "TRAINING_STAGES",
    "AttentionFunction",
    "EngineName",
    "QuantizationMethod",
    "RopeScaling",
]
