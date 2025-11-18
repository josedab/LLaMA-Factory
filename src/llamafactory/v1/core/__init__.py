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

"""Provide core engine components for v1 training infrastructure.

This module contains the foundational engine classes that power the v1 training
system. These engines handle the core responsibilities of data loading, model
initialization, and training orchestration, providing a clean abstraction layer
between configuration and execution.

The core module implements a separation of concerns pattern where each engine
is responsible for a specific aspect of the training pipeline, enabling better
testability, maintainability, and extensibility through plugins.

Key Classes:
    DataEngine: Unified data loading engine supporting multiple data sources,
        formats, and streaming modes.
    ModelEngine: Model loading and initialization engine with processor support.
    BaseTrainer: Abstract base trainer providing common training loop functionality.
    ChatSampler: Text generation sampler for inference workloads.
    DataCollator: Collates individual samples into batches for training.

Example:
    Create and use the data engine::

        from llamafactory.v1.config.parser import get_args
        from llamafactory.v1.core.data_engine import DataEngine

        data_args, *_ = get_args()
        data_engine = DataEngine(data_args)
        sample = data_engine[0]

    Initialize model and processor::

        from llamafactory.v1.core.model_engine import ModelEngine

        model_engine = ModelEngine(model_args)
        model = model_engine.get_model()
        processor = model_engine.get_processor()

See Also:
    llamafactory.v1.core.data_engine: Data loading implementation.
    llamafactory.v1.core.model_engine: Model loading implementation.
    llamafactory.v1.core.base_trainer: Base trainer implementation.
    llamafactory.v1.trainers: Specialized trainer implementations.
"""
