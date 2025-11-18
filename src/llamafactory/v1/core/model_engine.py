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

"""Implement model loading engine for v1 training.

This module provides the ModelEngine class which handles model and processor
loading for LLaMA Factory v1. It abstracts the complexity of loading various
model architectures and their associated tokenizers/processors.

The ModelEngine supports loading models from local paths and Hugging Face Hub,
with options for trust settings and custom configurations. It returns both the
model and processor needed for training and inference.

Key Classes:
    ModelEngine: Main model loading engine that provides methods to retrieve
        the model and processor based on configuration arguments.

Example:
    Load model and processor::

        from llamafactory.v1.config.model_args import ModelArguments
        from llamafactory.v1.core.model_engine import ModelEngine

        model_args = ModelArguments(
            model="meta-llama/Llama-2-7b-hf",
            trust_remote_code=True
        )
        engine = ModelEngine(model_args)
        model = engine.get_model()
        processor = engine.get_processor()

See Also:
    llamafactory.v1.config.model_args: Model configuration parameters.
    llamafactory.v1.plugins.model_plugins: Model customization plugins.
"""

from ..config.model_args import ModelArguments
from ..extras.types import Model, Processor


class ModelEngine:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args

    def get_model(self) -> Model:
        pass

    def get_processor(self) -> Processor:
        pass
