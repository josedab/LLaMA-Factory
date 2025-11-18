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

"""Provide data loading and conversion plugins for v1.

This module contains plugins for data loading, format conversion, indexing,
and template rendering. These plugins enable the v1 data engine to support
various data sources and formats while maintaining a consistent internal
representation.

Key Submodules:
    converter: Format converters (Alpaca, ShareGPT, pair) that transform raw
        data into standardized SFT/DPO samples.
    loader: Data loading plugins for files, directories, and Hugging Face Hub,
        plus index adjustment for dataset sizing and weighting.
    template: Message rendering templates for different model chat formats.

Key Classes:
    DataLoaderPlugin: Plugin for loading datasets from various file formats.
    DataIndexPlugin: Plugin for adjusting dataset indices by size or weight.
    DataSelectorPlugin: Plugin for selecting samples using various index types.

Key Functions:
    get_converter: Retrieve a registered format converter by name.
    alpaca_converter: Convert Alpaca format to SFTSample.
    sharegpt_converter: Convert ShareGPT format to SFTSample.
    pair_converter: Convert pair format to DPOSample.

Example:
    Convert Alpaca data to standard format::

        from llamafactory.v1.plugins.data_plugins.converter import get_converter

        converter = get_converter("alpaca")
        sample = converter({"instruction": "Hello", "output": "Hi!"})

    Load data from a JSON file::

        from llamafactory.v1.plugins.data_plugins.loader import DataLoaderPlugin

        loader = DataLoaderPlugin(data_args)
        dataset = loader.auto_load_data(dataset_info)

See Also:
    llamafactory.v1.core.data_engine: Main consumer of these plugins.
    llamafactory.v1.extras.types: Type definitions for samples.
"""
