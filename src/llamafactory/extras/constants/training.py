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

"""Training-related constants.

This module contains training stages, methods, and related enums
for configuring the training process.
"""

from enum import Enum, unique


# Training stages mapping (human-readable to internal)
TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}

# Stages that use paired data
STAGES_USE_PAIR_DATA = {"rm", "dpo"}

# Available fine-tuning methods
METHODS = ["full", "freeze", "lora", "oft"]

# Methods that use PEFT library
PEFT_METHODS = {"lora", "oft"}


class AttentionFunction(str, Enum):
    """Attention implementation options."""

    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"


class EngineName(str, Enum):
    """Inference engine options."""

    HF = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"
    KT = "ktransformers"


@unique
class QuantizationMethod(str, Enum):
    """Quantization method options.

    Borrowed from `transformers.utils.quantization_config.QuantizationMethod`.
    """

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"
    MXFP4 = "mxfp4"


class RopeScaling(str, Enum):
    """RoPE scaling method options."""

    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


__all__ = [
    "TRAINING_STAGES",
    "STAGES_USE_PAIR_DATA",
    "METHODS",
    "PEFT_METHODS",
    "AttentionFunction",
    "EngineName",
    "QuantizationMethod",
    "RopeScaling",
]
