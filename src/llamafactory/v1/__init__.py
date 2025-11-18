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

"""Provide the next-generation v1 training framework for LLaMA Factory.

This module contains the redesigned v1 architecture for LLaMA Factory, offering
a more modular and extensible approach to language model training. The v1 system
introduces a plugin-based architecture for data loading, model customization,
and distributed training, enabling better separation of concerns and easier
customization.

The v1 module is organized into several submodules:
    - config: Configuration dataclasses for data, model, sampling, and training
    - core: Core components including engines and base trainer
    - extras: Type definitions and utilities
    - plugins: Extensible plugins for data, models, samplers, and trainers
    - trainers: Specialized trainers for different training paradigms

Key Classes:
    DataEngine: Unified data loading and preprocessing engine.
    ModelEngine: Model loading and initialization engine.
    BaseTrainer: Base class for all training implementations.
    SFTTrainer: Supervised fine-tuning trainer.

Key Functions:
    launch: Main entry point for the v1 CLI.
    get_args: Parse configuration arguments from CLI or config files.
    run_sft: Run supervised fine-tuning training.

Example:
    Launch v1 training from command line::

        llamafactory-cli sft config.yaml

    Or programmatically::

        from llamafactory.v1.trainers.sft_trainer import run_sft
        run_sft()

See Also:
    llamafactory.v1.launcher: CLI launcher implementation.
    llamafactory.v1.config: Configuration argument definitions.
    llamafactory.v1.trainers: Trainer implementations for different paradigms.
"""
