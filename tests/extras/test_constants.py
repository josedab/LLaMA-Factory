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

"""Tests for the refactored constants module.

This module tests backwards compatibility, submodule imports,
and correct constant definitions.
"""

import pytest


class TestBackwardsCompatibility:
    """Test that existing import patterns still work."""

    def test_import_from_constants(self):
        """Test importing from the main constants module."""
        from llamafactory.extras.constants import (
            SUPPORTED_MODELS,
            IGNORE_INDEX,
            TRAINING_STAGES,
            CHECKPOINT_NAMES,
            FILEEXT2TYPE,
            DEFAULT_TEMPLATE,
            METHODS,
            PEFT_METHODS,
        )

        assert isinstance(SUPPORTED_MODELS, dict)
        assert IGNORE_INDEX == -100
        assert isinstance(TRAINING_STAGES, dict)
        assert isinstance(CHECKPOINT_NAMES, set)
        assert isinstance(FILEEXT2TYPE, dict)
        assert isinstance(DEFAULT_TEMPLATE, dict)
        assert isinstance(METHODS, list)
        assert isinstance(PEFT_METHODS, set)

    def test_import_enums(self):
        """Test importing enum classes."""
        from llamafactory.extras.constants import (
            DownloadSource,
            AttentionFunction,
            EngineName,
            QuantizationMethod,
            RopeScaling,
        )

        # Test enum values
        assert DownloadSource.DEFAULT.value == "hf"
        assert AttentionFunction.AUTO.value == "auto"
        assert EngineName.HF.value == "huggingface"
        assert QuantizationMethod.BNB.value == "bnb"
        assert RopeScaling.LINEAR.value == "linear"

    def test_import_function(self):
        """Test importing the register_model_group function."""
        from llamafactory.extras.constants import register_model_group

        assert callable(register_model_group)


class TestSubmoduleImports:
    """Test importing from specific submodules."""

    def test_import_from_models(self):
        """Test importing from the models submodule."""
        from llamafactory.extras.constants.models import (
            SUPPORTED_MODELS,
            DEFAULT_TEMPLATE,
            MULTIMODAL_SUPPORTED_MODELS,
            MCA_SUPPORTED_MODELS,
            MOD_SUPPORTED_MODELS,
            SUPPORTED_CLASS_FOR_S2ATTN,
            DownloadSource,
            register_model_group,
        )

        assert isinstance(SUPPORTED_MODELS, dict)
        assert len(SUPPORTED_MODELS) > 0  # Should have models registered
        assert isinstance(MCA_SUPPORTED_MODELS, set)
        assert "llama" in MCA_SUPPORTED_MODELS

    def test_import_from_training(self):
        """Test importing from the training submodule."""
        from llamafactory.extras.constants.training import (
            TRAINING_STAGES,
            STAGES_USE_PAIR_DATA,
            METHODS,
            PEFT_METHODS,
            AttentionFunction,
            EngineName,
            QuantizationMethod,
            RopeScaling,
        )

        assert "sft" in TRAINING_STAGES.values()
        assert "rm" in STAGES_USE_PAIR_DATA
        assert "lora" in METHODS
        assert "lora" in PEFT_METHODS

    def test_import_from_data(self):
        """Test importing from the data submodule."""
        from llamafactory.extras.constants.data import (
            FILEEXT2TYPE,
            IGNORE_INDEX,
            CHOICES,
            SUBJECTS,
        )

        assert FILEEXT2TYPE["json"] == "json"
        assert FILEEXT2TYPE["jsonl"] == "json"
        assert IGNORE_INDEX == -100
        assert len(CHOICES) == 4
        assert "Average" in SUBJECTS

    def test_import_from_defaults(self):
        """Test importing from the defaults submodule."""
        from llamafactory.extras.constants.defaults import (
            CHECKPOINT_NAMES,
            RUNNING_LOG,
            TRAINER_LOG,
            LLAMABOARD_CONFIG,
            DATA_CONFIG,
            V_HEAD_WEIGHTS_NAME,
            LAYERNORM_NAMES,
            AUDIO_PLACEHOLDER,
            IMAGE_PLACEHOLDER,
            VIDEO_PLACEHOLDER,
        )

        assert isinstance(CHECKPOINT_NAMES, set)
        assert RUNNING_LOG == "running_log.txt"
        assert TRAINER_LOG == "trainer_log.jsonl"
        assert "norm" in LAYERNORM_NAMES

    def test_import_from_templates(self):
        """Test importing from the templates submodule."""
        from llamafactory.extras.constants.templates import DEFAULT_TEMPLATE, get_template

        assert isinstance(DEFAULT_TEMPLATE, dict)
        assert callable(get_template)


class TestModelsModule:
    """Test the models module functionality."""

    def test_supported_models_populated(self):
        """Test that SUPPORTED_MODELS is populated with models."""
        from llamafactory.extras.constants.models import SUPPORTED_MODELS

        # Should have many models registered
        assert len(SUPPORTED_MODELS) > 100

    def test_download_source_enum(self):
        """Test DownloadSource enum values."""
        from llamafactory.extras.constants.models import DownloadSource

        assert DownloadSource.DEFAULT.value == "hf"
        assert DownloadSource.MODELSCOPE.value == "ms"
        assert DownloadSource.OPENMIND.value == "om"

    def test_model_has_download_sources(self):
        """Test that models have proper download source structure."""
        from llamafactory.extras.constants.models import SUPPORTED_MODELS, DownloadSource

        # Check a few known models
        for model_name, sources in list(SUPPORTED_MODELS.items())[:5]:
            assert isinstance(sources, dict)
            # Each source should be a DownloadSource key
            for source_key in sources.keys():
                assert isinstance(source_key, DownloadSource)

    def test_default_template_populated(self):
        """Test that DEFAULT_TEMPLATE is populated for chat models."""
        from llamafactory.extras.constants.models import DEFAULT_TEMPLATE, SUPPORTED_MODELS

        # Find models with -Chat suffix that should have templates
        chat_models = [name for name in SUPPORTED_MODELS.keys() if "-Chat" in name]
        assert len(chat_models) > 0

        # At least some chat models should have templates
        chat_models_with_templates = [name for name in chat_models if name in DEFAULT_TEMPLATE]
        assert len(chat_models_with_templates) > 0

    def test_multimodal_models(self):
        """Test that multimodal models are tracked."""
        from llamafactory.extras.constants.models import MULTIMODAL_SUPPORTED_MODELS

        # Should have some multimodal models
        assert len(MULTIMODAL_SUPPORTED_MODELS) > 0


class TestTrainingModule:
    """Test the training module constants."""

    def test_training_stages(self):
        """Test training stages mapping."""
        from llamafactory.extras.constants.training import TRAINING_STAGES

        expected_stages = ["sft", "rm", "ppo", "dpo", "kto", "pt"]
        for stage in expected_stages:
            assert stage in TRAINING_STAGES.values()

    def test_methods(self):
        """Test available methods."""
        from llamafactory.extras.constants.training import METHODS

        assert "full" in METHODS
        assert "freeze" in METHODS
        assert "lora" in METHODS
        assert "oft" in METHODS

    def test_peft_methods_subset(self):
        """Test that PEFT methods are a subset of all methods."""
        from llamafactory.extras.constants.training import METHODS, PEFT_METHODS

        for method in PEFT_METHODS:
            assert method in METHODS

    def test_attention_function_enum(self):
        """Test AttentionFunction enum."""
        from llamafactory.extras.constants.training import AttentionFunction

        assert AttentionFunction.AUTO.value == "auto"
        assert AttentionFunction.FA2.value == "fa2"
        assert AttentionFunction.SDPA.value == "sdpa"

    def test_quantization_methods(self):
        """Test QuantizationMethod enum."""
        from llamafactory.extras.constants.training import QuantizationMethod

        expected_methods = ["bnb", "gptq", "awq", "aqlm", "quanto", "eetq", "hqq"]
        for method in expected_methods:
            assert method in [m.value for m in QuantizationMethod]


class TestDataModule:
    """Test the data module constants."""

    def test_file_extensions(self):
        """Test file extension mappings."""
        from llamafactory.extras.constants.data import FILEEXT2TYPE

        assert FILEEXT2TYPE["json"] == "json"
        assert FILEEXT2TYPE["jsonl"] == "json"
        assert FILEEXT2TYPE["csv"] == "csv"
        assert FILEEXT2TYPE["parquet"] == "parquet"
        assert FILEEXT2TYPE["arrow"] == "arrow"
        assert FILEEXT2TYPE["txt"] == "text"

    def test_ignore_index(self):
        """Test IGNORE_INDEX value."""
        from llamafactory.extras.constants.data import IGNORE_INDEX

        assert IGNORE_INDEX == -100


class TestDefaultsModule:
    """Test the defaults module constants."""

    def test_log_files(self):
        """Test log file name constants."""
        from llamafactory.extras.constants.defaults import RUNNING_LOG, TRAINER_LOG

        assert RUNNING_LOG.endswith(".txt")
        assert TRAINER_LOG.endswith(".jsonl")

    def test_config_files(self):
        """Test config file name constants."""
        from llamafactory.extras.constants.defaults import (
            DATA_CONFIG,
            LLAMABOARD_CONFIG,
            TRAINING_ARGS,
        )

        assert DATA_CONFIG.endswith(".json")
        assert LLAMABOARD_CONFIG.endswith(".yaml")
        assert TRAINING_ARGS.endswith(".yaml")

    def test_placeholders(self):
        """Test multimodal placeholder constants."""
        from llamafactory.extras.constants.defaults import (
            AUDIO_PLACEHOLDER,
            IMAGE_PLACEHOLDER,
            VIDEO_PLACEHOLDER,
        )

        # Should be non-empty strings
        assert len(AUDIO_PLACEHOLDER) > 0
        assert len(IMAGE_PLACEHOLDER) > 0
        assert len(VIDEO_PLACEHOLDER) > 0


class TestTemplatesModule:
    """Test the templates module functionality."""

    def test_get_template_function(self):
        """Test the get_template helper function."""
        from llamafactory.extras.constants.templates import get_template, DEFAULT_TEMPLATE

        # Test getting template for a known chat model
        for model_name in DEFAULT_TEMPLATE:
            template = get_template(model_name)
            assert isinstance(template, str)
            break  # Just test one

    def test_get_template_unknown_model(self):
        """Test get_template returns empty string for unknown model."""
        from llamafactory.extras.constants.templates import get_template

        result = get_template("NonexistentModel-12345")
        assert result == ""


class TestAllExports:
    """Test that __all__ exports are correct."""

    def test_main_module_all(self):
        """Test that __all__ in main module matches actual exports."""
        from llamafactory.extras import constants

        for name in constants.__all__:
            assert hasattr(constants, name), f"Missing export: {name}"

    def test_no_missing_constants(self):
        """Test that key constants are not missing from the refactored module."""
        from llamafactory.extras.constants import (
            # Critical constants that must be present
            SUPPORTED_MODELS,
            DEFAULT_TEMPLATE,
            IGNORE_INDEX,
            TRAINING_STAGES,
            METHODS,
            PEFT_METHODS,
            CHECKPOINT_NAMES,
            FILEEXT2TYPE,
            DownloadSource,
            AttentionFunction,
            EngineName,
            QuantizationMethod,
            RopeScaling,
        )

        # All imports should succeed
        assert SUPPORTED_MODELS is not None
        assert DEFAULT_TEMPLATE is not None
