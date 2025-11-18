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

"""Implement Supervised Fine-Tuning trainer for v1.

This module provides the SFTTrainer class for training language models using
supervised fine-tuning on instruction-response data. SFT is the foundation for
teaching models to follow instructions and produce helpful responses.

Key Classes:
    SFTTrainer: Trainer class extending BaseTrainer with SFT-specific
        loss computation and training loop.

Key Functions:
    run_sft: Entry point function that sets up and runs the complete SFT
        training pipeline including data loading, model initialization,
        and trainer execution.

Example:
    Run SFT training from command line::

        from llamafactory.v1.trainers.sft_trainer import run_sft
        run_sft()

    Use SFTTrainer directly::

        from llamafactory.v1.config.parser import get_args
        from llamafactory.v1.core.data_engine import DataEngine
        from llamafactory.v1.core.model_engine import ModelEngine
        from llamafactory.v1.trainers.sft_trainer import SFTTrainer

        model_args, data_args, training_args, _ = get_args()
        model_engine = ModelEngine(model_args)
        data_engine = DataEngine(data_args)

        trainer = SFTTrainer(
            training_args,
            model_engine.get_model(),
            model_engine.get_processor(),
            data_engine.get_data_loader(model_engine.get_processor())
        )
        trainer.fit()

See Also:
    llamafactory.v1.core.base_trainer: Base trainer class.
    llamafactory.v1.core.data_engine: Data loading engine.
    llamafactory.v1.core.model_engine: Model loading engine.
    llamafactory.v1.config.training_args: Training configuration.
"""

from ..config.parser import get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine


class SFTTrainer(BaseTrainer):
    pass


def run_sft():
    model_args, data_args, training_args, _ = get_args()
    model_engine = ModelEngine(model_args)
    data_engine = DataEngine(data_args)
    model = model_engine.get_model()
    processor = model_engine.get_processor()
    data_loader = data_engine.get_data_loader(processor)
    trainer = SFTTrainer(training_args, model, processor, data_loader)
    trainer.fit()
