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

r"""LLaMA-Factory: Efficient fine-tuning of large language models.

This package provides a unified framework for fine-tuning large language
models with various techniques including LoRA, QLoRA, full fine-tuning,
and more. It supports multiple model architectures and training paradigms.

Module Hierarchy:
    api, webui > chat, eval, train > data, model > hparams > extras

Key Submodules:
    cli: Command-line interface entry point
    launcher: Command routing and distributed training setup
    train: Training orchestration and model tuning
    data: Dataset loading and preprocessing
    model: Model loading and configuration
    api: OpenAI-compatible API server
    chat: Interactive chat interfaces
    webui: Web-based user interfaces (LlamaBoard)
    hparams: Hyperparameter management
    extras: Utilities, logging, and environment helpers

Environment Variables:
    DISABLE_VERSION_CHECK=1: Disable version checking
    RECORD_VRAM=1: Enable VRAM recording
    FORCE_TORCHRUN=1: Force using torchrun
    LLAMAFACTORY_VERBOSITY=WARN: Set logging verbosity
    USE_MODELSCOPE_HUB=1: Use ModelScope hub
    USE_OPENMIND_HUB=1: Use OpenMind hub
    USE_V1=1: Use V1 API launcher

Example:
    Basic usage::

        import llamafactory
        print(llamafactory.__version__)

    Command line usage::

        llamafactory-cli train config.yaml
        llamafactory-cli webui

See Also:
    - cli: Main CLI entry point
    - launcher: Command dispatcher
    - train.tuner: Training orchestration
    - GitHub: https://github.com/hiyouga/LLaMA-Factory
"""

from .extras.env import VERSION


__version__ = VERSION
