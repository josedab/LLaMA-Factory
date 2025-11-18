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

"""Tests for type definitions and Protocol implementations."""

from __future__ import annotations

import pytest


class TestTypeImports:
    """Test that type definitions can be imported correctly."""

    def test_hparams_types_import(self) -> None:
        """Test that hparams type definitions can be imported."""
        from llamafactory.hparams import (
            DataArgsType,
            EvalArgsTuple,
            EvaluationArgsType,
            FinetuningArgsType,
            GeneratingArgsType,
            InferArgsTuple,
            ModelArgsType,
            RayArgsType,
            TrainArgsTuple,
            TrainingArgsType,
        )

        # Verify type variables exist
        assert DataArgsType is not None
        assert EvaluationArgsType is not None
        assert FinetuningArgsType is not None
        assert GeneratingArgsType is not None
        assert ModelArgsType is not None
        assert RayArgsType is not None
        assert TrainingArgsType is not None

        # Verify type aliases exist
        assert TrainArgsTuple is not None
        assert InferArgsTuple is not None
        assert EvalArgsTuple is not None

    def test_chat_protocols_import(self) -> None:
        """Test that chat Protocol definitions can be imported."""
        from llamafactory.chat import (
            Generatable,
            InferenceEngine,
            Tokenizable,
        )

        # Verify protocols exist
        assert InferenceEngine is not None
        assert Tokenizable is not None
        assert Generatable is not None

    def test_hparams_classes_import(self) -> None:
        """Test that hparams dataclasses can be imported."""
        from llamafactory.hparams import (
            DataArguments,
            EvaluationArguments,
            FinetuningArguments,
            GeneratingArguments,
            ModelArguments,
            RayArguments,
            TrainingArguments,
        )

        # Verify classes exist
        assert DataArguments is not None
        assert EvaluationArguments is not None
        assert FinetuningArguments is not None
        assert GeneratingArguments is not None
        assert ModelArguments is not None
        assert RayArguments is not None
        assert TrainingArguments is not None


class TestProtocolImplementations:
    """Test that Protocol implementations work correctly."""

    def test_inference_engine_protocol_is_runtime_checkable(self) -> None:
        """Test that InferenceEngine is runtime checkable."""
        from llamafactory.chat import InferenceEngine

        # Verify it's a Protocol
        assert hasattr(InferenceEngine, "__protocol_attrs__")

    def test_tokenizable_protocol_is_runtime_checkable(self) -> None:
        """Test that Tokenizable is runtime checkable."""
        from llamafactory.chat import Tokenizable

        # Verify it's a Protocol
        assert hasattr(Tokenizable, "__protocol_attrs__")

    def test_generatable_protocol_is_runtime_checkable(self) -> None:
        """Test that Generatable is runtime checkable."""
        from llamafactory.chat import Generatable

        # Verify it's a Protocol
        assert hasattr(Generatable, "__protocol_attrs__")


class TestExtrasTyping:
    """Test that extras module typing works correctly."""

    def test_packages_functions_return_bool(self) -> None:
        """Test that package availability functions return bool."""
        from llamafactory.extras.packages import (
            is_fastapi_available,
            is_gradio_available,
            is_matplotlib_available,
            is_pyav_available,
            is_vllm_available,
        )

        # Verify functions return bool
        assert isinstance(is_pyav_available(), bool)
        assert isinstance(is_fastapi_available(), bool)
        assert isinstance(is_gradio_available(), bool)
        assert isinstance(is_matplotlib_available(), bool)
        assert isinstance(is_vllm_available(), bool)

    def test_misc_functions_return_correct_types(self) -> None:
        """Test that misc utility functions have correct return types."""
        from llamafactory.extras.misc import (
            get_current_device,
            get_device_count,
            is_env_enabled,
        )

        import torch

        # Verify return types
        device = get_current_device()
        assert isinstance(device, torch.device)

        device_count = get_device_count()
        assert isinstance(device_count, int)

        env_enabled = is_env_enabled("TEST_VAR", "0")
        assert isinstance(env_enabled, bool)


class TestTypeAliases:
    """Test that type aliases work correctly with actual types."""

    def test_train_args_tuple_structure(self) -> None:
        """Test that TrainArgsTuple matches expected structure."""
        from llamafactory.hparams import (
            DataArguments,
            FinetuningArguments,
            GeneratingArguments,
            ModelArguments,
            TrainingArguments,
        )

        # Create instances to verify structure
        model_args = ModelArguments()
        data_args = DataArguments()
        training_args = TrainingArguments(output_dir="test")
        finetuning_args = FinetuningArguments()
        generating_args = GeneratingArguments()

        # Verify tuple can be created
        args_tuple = (model_args, data_args, training_args, finetuning_args, generating_args)
        assert len(args_tuple) == 5
        assert isinstance(args_tuple[0], ModelArguments)
        assert isinstance(args_tuple[1], DataArguments)
        assert isinstance(args_tuple[2], TrainingArguments)
        assert isinstance(args_tuple[3], FinetuningArguments)
        assert isinstance(args_tuple[4], GeneratingArguments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
