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

"""Provide extensible plugin system for v1 customization.

This module serves as the root package for the v1 plugin architecture, which
enables customization and extension of data processing, model optimization,
sampling, and training components. The plugin system follows a registry-based
pattern that allows users to add new functionality without modifying core code.

The plugins are organized into four main categories:
    - data_plugins: Data loading, conversion, and template formatting
    - model_plugins: Model customization including kernels and PEFT
    - sampler_plugins: Text generation and sampling backends
    - trainer_plugins: Distributed training and optimization

Key Submodules:
    data_plugins: Converters for data formats, loaders for various sources,
        and template systems for formatting.
    model_plugins: Hardware-optimized kernels (NPU, CUDA), tokenizer extensions,
        and parameter-efficient fine-tuning.
    sampler_plugins: vLLM and other inference backends.
    trainer_plugins: Distributed training with Accelerate and other backends.

Example:
    Register a custom data converter::

        from llamafactory.v1.plugins.data_plugins.converter import CONVERTERS

        def my_converter(raw_sample):
            return {"messages": [...]}

        CONVERTERS["my_format"] = my_converter

    Apply a model kernel::

        from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
        from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import NpuRMSNormKernel

        model = apply_kernel(model, NpuRMSNormKernel)

See Also:
    llamafactory.v1.plugins.data_plugins: Data processing plugins.
    llamafactory.v1.plugins.model_plugins: Model optimization plugins.
    llamafactory.v1.plugins.sampler_plugins: Inference backend plugins.
    llamafactory.v1.plugins.trainer_plugins: Training backend plugins.
"""
