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

"""Tests for the custom exception hierarchy."""

import pytest

from llamafactory.exceptions import (
    ERROR_CODES,
    LLaMAFactoryError,
    # Configuration errors
    ConfigurationError,
    InvalidArgumentError,
    IncompatibleArgumentsError,
    MissingArgumentError,
    # Model errors
    ModelError,
    ModelLoadingError,
    TokenizerError,
    AdapterError,
    IncompatibleModelError,
    # Data errors
    DataError,
    DatasetNotFoundError,
    DataFormatError,
    DataProcessingError,
    # Training errors
    TrainingError,
    CheckpointError,
    DistributedTrainingError,
    # Inference errors
    InferenceError,
    EngineError,
    GenerationError,
    # Dependency errors
    DependencyError,
    MissingDependencyError,
    IncompatibleVersionError,
)


class TestLLaMAFactoryError:
    """Tests for the base LLaMAFactoryError class."""

    def test_basic_exception(self):
        """Test basic exception creation with message only."""
        error = LLaMAFactoryError("Test error message")
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
        assert error.hint is None
        assert str(error) == "Test error message"

    def test_exception_with_error_code(self):
        """Test exception with error code."""
        error = LLaMAFactoryError("Test error", error_code="LF100")
        assert error.error_code == "LF100"
        assert "[LF100]" in str(error)
        assert "Test error" in str(error)

    def test_exception_with_context(self):
        """Test exception with context dictionary."""
        context = {"key": "value", "count": 42}
        error = LLaMAFactoryError("Test error", context=context)
        assert error.context == context
        assert error.context["key"] == "value"
        assert error.context["count"] == 42

    def test_exception_with_hint(self):
        """Test exception with hint message."""
        error = LLaMAFactoryError("Test error", hint="Try doing X instead")
        assert error.hint == "Try doing X instead"
        assert "Hint: Try doing X instead" in str(error)

    def test_exception_with_all_fields(self):
        """Test exception with all fields populated."""
        error = LLaMAFactoryError(
            message="Complete test error",
            error_code="LF999",
            context={"param": "value"},
            hint="This is a helpful hint",
        )
        assert error.message == "Complete test error"
        assert error.error_code == "LF999"
        assert error.context == {"param": "value"}
        assert error.hint == "This is a helpful hint"

        error_str = str(error)
        assert "[LF999]" in error_str
        assert "Complete test error" in error_str
        assert "Hint: This is a helpful hint" in error_str

    def test_exception_repr(self):
        """Test exception repr for debugging."""
        error = LLaMAFactoryError(
            message="Test",
            error_code="LF001",
            context={"key": "value"},
            hint="Hint",
        )
        repr_str = repr(error)
        assert "LLaMAFactoryError" in repr_str
        assert "message='Test'" in repr_str
        assert "error_code='LF001'" in repr_str

    def test_exception_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(LLaMAFactoryError) as exc_info:
            raise LLaMAFactoryError("Test error")
        assert str(exc_info.value) == "Test error"


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from ValueError for backwards compatibility."""
        error = ConfigurationError("Config error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, ValueError)

    def test_invalid_argument_error(self):
        """Test InvalidArgumentError."""
        error = InvalidArgumentError(
            message="Invalid batch size",
            error_code="LF100",
            context={"batch_size": -1},
            hint="Batch size must be positive",
        )
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, ValueError)

        with pytest.raises(InvalidArgumentError):
            raise error

    def test_incompatible_arguments_error(self):
        """Test IncompatibleArgumentsError."""
        error = IncompatibleArgumentsError(
            message="Adapter is only valid for LoRA",
            error_code="LF101",
            context={
                "adapter_name_or_path": "my_adapter",
                "finetuning_type": "full",
            },
            hint="Set finetuning_type='lora'",
        )
        assert isinstance(error, ConfigurationError)

        with pytest.raises(IncompatibleArgumentsError):
            raise error

    def test_missing_argument_error(self):
        """Test MissingArgumentError."""
        error = MissingArgumentError(
            message="model_name_or_path is required",
            error_code="LF102",
        )
        assert isinstance(error, ConfigurationError)

        with pytest.raises(MissingArgumentError):
            raise error

    def test_catch_configuration_error_as_valueerror(self):
        """Test backwards compatibility - catching as ValueError."""
        with pytest.raises(ValueError):
            raise ConfigurationError("Test error")


class TestModelErrors:
    """Tests for model-related exceptions."""

    def test_model_error_inheritance(self):
        """Test ModelError inherits from RuntimeError for backwards compatibility."""
        error = ModelError("Model error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, RuntimeError)

    def test_model_loading_error(self):
        """Test ModelLoadingError."""
        error = ModelLoadingError(
            message="Failed to load model",
            error_code="LF201",
            context={"model_path": "/path/to/model"},
            hint="Check if the model path exists",
        )
        assert isinstance(error, ModelError)

        with pytest.raises(ModelLoadingError):
            raise error

    def test_tokenizer_error(self):
        """Test TokenizerError."""
        error = TokenizerError(
            message="Failed to load tokenizer",
            error_code="LF202",
        )
        assert isinstance(error, ModelError)

        with pytest.raises(TokenizerError):
            raise error

    def test_adapter_error(self):
        """Test AdapterError."""
        error = AdapterError(
            message="Adapter initialization failed",
            error_code="LF203",
            context={"adapter_type": "lora"},
        )
        assert isinstance(error, ModelError)

        with pytest.raises(AdapterError):
            raise error

    def test_incompatible_model_error(self):
        """Test IncompatibleModelError."""
        error = IncompatibleModelError(
            message="Model architecture not supported",
            error_code="LF204",
        )
        assert isinstance(error, ModelError)

        with pytest.raises(IncompatibleModelError):
            raise error


class TestDataErrors:
    """Tests for data-related exceptions."""

    def test_data_error_inheritance(self):
        """Test DataError inherits from ValueError."""
        error = DataError("Data error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, ValueError)

    def test_dataset_not_found_error(self):
        """Test DatasetNotFoundError."""
        error = DatasetNotFoundError(
            message="Dataset 'my_dataset' not found",
            error_code="LF300",
            context={"dataset_name": "my_dataset"},
            hint="Check dataset_info.json for available datasets",
        )
        assert isinstance(error, DataError)
        assert isinstance(error, FileNotFoundError)

        with pytest.raises(DatasetNotFoundError):
            raise error

    def test_data_format_error(self):
        """Test DataFormatError."""
        error = DataFormatError(
            message="Invalid JSON format",
            error_code="LF301",
        )
        assert isinstance(error, DataError)

        with pytest.raises(DataFormatError):
            raise error

    def test_data_processing_error(self):
        """Test DataProcessingError."""
        error = DataProcessingError(
            message="Tokenization failed",
            error_code="LF302",
        )
        assert isinstance(error, DataError)

        with pytest.raises(DataProcessingError):
            raise error


class TestTrainingErrors:
    """Tests for training-related exceptions."""

    def test_training_error_inheritance(self):
        """Test TrainingError inherits from RuntimeError."""
        error = TrainingError("Training error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, RuntimeError)

    def test_checkpoint_error(self):
        """Test CheckpointError."""
        error = CheckpointError(
            message="Failed to load checkpoint",
            error_code="LF401",
            context={"checkpoint_path": "/path/to/checkpoint"},
        )
        assert isinstance(error, TrainingError)

        with pytest.raises(CheckpointError):
            raise error

    def test_distributed_training_error(self):
        """Test DistributedTrainingError."""
        error = DistributedTrainingError(
            message="Distributed setup failed",
            error_code="LF403",
            hint="Check NCCL configuration",
        )
        assert isinstance(error, TrainingError)

        with pytest.raises(DistributedTrainingError):
            raise error


class TestInferenceErrors:
    """Tests for inference-related exceptions."""

    def test_inference_error_inheritance(self):
        """Test InferenceError inherits from RuntimeError."""
        error = InferenceError("Inference error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, RuntimeError)

    def test_engine_error(self):
        """Test EngineError."""
        error = EngineError(
            message="Engine initialization failed",
            error_code="LF500",
        )
        assert isinstance(error, InferenceError)

        with pytest.raises(EngineError):
            raise error

    def test_generation_error(self):
        """Test GenerationError."""
        error = GenerationError(
            message="Text generation failed",
            error_code="LF501",
            context={"max_length": 1024},
        )
        assert isinstance(error, InferenceError)

        with pytest.raises(GenerationError):
            raise error


class TestDependencyErrors:
    """Tests for dependency-related exceptions."""

    def test_dependency_error_inheritance(self):
        """Test DependencyError inherits from ImportError."""
        error = DependencyError("Dependency error")
        assert isinstance(error, LLaMAFactoryError)
        assert isinstance(error, ImportError)

    def test_missing_dependency_error(self):
        """Test MissingDependencyError."""
        error = MissingDependencyError(
            message="vllm is not installed",
            error_code="LF600",
            hint="Install with: pip install vllm",
        )
        assert isinstance(error, DependencyError)

        with pytest.raises(MissingDependencyError):
            raise error

    def test_incompatible_version_error(self):
        """Test IncompatibleVersionError."""
        error = IncompatibleVersionError(
            message="transformers version incompatible",
            error_code="LF601",
            context={"required": ">=4.40.0", "installed": "4.39.0"},
        )
        assert isinstance(error, DependencyError)

        with pytest.raises(IncompatibleVersionError):
            raise error


class TestErrorCodes:
    """Tests for ERROR_CODES dictionary."""

    def test_error_codes_not_empty(self):
        """Test that ERROR_CODES is not empty."""
        assert len(ERROR_CODES) > 0

    def test_error_codes_format(self):
        """Test that all error codes follow the correct format."""
        for code, description in ERROR_CODES.items():
            assert code.startswith("LF"), f"Code {code} should start with 'LF'"
            assert len(code) == 5, f"Code {code} should be 5 characters"
            assert code[2:].isdigit(), f"Code {code} should have digits after 'LF'"
            assert isinstance(description, str), f"Description for {code} should be string"
            assert len(description) > 0, f"Description for {code} should not be empty"

    def test_configuration_error_codes(self):
        """Test configuration error codes (LF1xx)."""
        config_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF1")]
        assert len(config_codes) >= 4
        assert "LF100" in ERROR_CODES
        assert "LF101" in ERROR_CODES
        assert "LF102" in ERROR_CODES
        assert "LF103" in ERROR_CODES

    def test_model_error_codes(self):
        """Test model error codes (LF2xx)."""
        model_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF2")]
        assert len(model_codes) >= 5
        assert "LF200" in ERROR_CODES
        assert "LF201" in ERROR_CODES
        assert "LF202" in ERROR_CODES
        assert "LF203" in ERROR_CODES
        assert "LF204" in ERROR_CODES

    def test_data_error_codes(self):
        """Test data error codes (LF3xx)."""
        data_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF3")]
        assert len(data_codes) >= 4
        assert "LF300" in ERROR_CODES
        assert "LF301" in ERROR_CODES
        assert "LF302" in ERROR_CODES
        assert "LF303" in ERROR_CODES

    def test_training_error_codes(self):
        """Test training error codes (LF4xx)."""
        training_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF4")]
        assert len(training_codes) >= 4
        assert "LF400" in ERROR_CODES
        assert "LF401" in ERROR_CODES
        assert "LF402" in ERROR_CODES
        assert "LF403" in ERROR_CODES

    def test_inference_error_codes(self):
        """Test inference error codes (LF5xx)."""
        inference_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF5")]
        assert len(inference_codes) >= 3
        assert "LF500" in ERROR_CODES
        assert "LF501" in ERROR_CODES
        assert "LF502" in ERROR_CODES

    def test_dependency_error_codes(self):
        """Test dependency error codes (LF6xx)."""
        dep_codes = [c for c in ERROR_CODES.keys() if c.startswith("LF6")]
        assert len(dep_codes) >= 2
        assert "LF600" in ERROR_CODES
        assert "LF601" in ERROR_CODES


class TestExceptionHierarchy:
    """Tests for exception hierarchy and inheritance."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from LLaMAFactoryError."""
        exceptions = [
            ConfigurationError,
            InvalidArgumentError,
            IncompatibleArgumentsError,
            MissingArgumentError,
            ModelError,
            ModelLoadingError,
            TokenizerError,
            AdapterError,
            IncompatibleModelError,
            DataError,
            DatasetNotFoundError,
            DataFormatError,
            DataProcessingError,
            TrainingError,
            CheckpointError,
            DistributedTrainingError,
            InferenceError,
            EngineError,
            GenerationError,
            DependencyError,
            MissingDependencyError,
            IncompatibleVersionError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, LLaMAFactoryError)

    def test_catch_all_llamafactory_errors(self):
        """Test that all exceptions can be caught by catching LLaMAFactoryError."""
        exceptions_to_test = [
            ConfigurationError("test"),
            InvalidArgumentError("test"),
            ModelLoadingError("test"),
            DatasetNotFoundError("test"),
            CheckpointError("test"),
            EngineError("test"),
            MissingDependencyError("test"),
        ]

        for error in exceptions_to_test:
            with pytest.raises(LLaMAFactoryError):
                raise error

    def test_programmatic_error_handling(self):
        """Test programmatic error handling example from RFC."""
        def simulate_error(error_type: str):
            if error_type == "model":
                raise ModelLoadingError(
                    message="Model not found",
                    error_code="LF201",
                    context={"model": "test"},
                )
            elif error_type == "data":
                raise DatasetNotFoundError(
                    message="Dataset not found",
                    error_code="LF300",
                )
            else:
                raise LLaMAFactoryError("Unknown error")

        # Test specific error handling
        with pytest.raises(ModelLoadingError) as exc_info:
            simulate_error("model")
        assert exc_info.value.error_code == "LF201"

        # Test catching parent type
        with pytest.raises(ModelError):
            simulate_error("model")

        # Test catching base type
        with pytest.raises(LLaMAFactoryError):
            simulate_error("data")


class TestUsagePatterns:
    """Tests for common usage patterns from RFC examples."""

    def test_example_from_rfc(self):
        """Test the example usage pattern from the RFC."""
        # This mirrors the "After" example in the RFC
        adapter_name_or_path = "my_adapter"
        finetuning_type = "full"

        with pytest.raises(IncompatibleArgumentsError) as exc_info:
            if adapter_name_or_path and finetuning_type != "lora":
                raise IncompatibleArgumentsError(
                    message="Adapter is only valid for the LoRA method",
                    error_code="LF101",
                    context={
                        "adapter_name_or_path": adapter_name_or_path,
                        "finetuning_type": finetuning_type,
                    },
                    hint="Either remove adapter_name_or_path or set finetuning_type='lora'",
                )

        error = exc_info.value
        assert error.error_code == "LF101"
        assert error.context["adapter_name_or_path"] == "my_adapter"
        assert error.context["finetuning_type"] == "full"
        assert "LoRA" in str(error)
        assert "Hint:" in str(error)

    def test_error_without_optional_fields(self):
        """Test that exceptions work correctly without optional fields."""
        error = InvalidArgumentError("Simple error message")
        assert error.message == "Simple error message"
        assert error.error_code is None
        assert error.context == {}
        assert error.hint is None
        assert str(error) == "Simple error message"

    def test_context_modification(self):
        """Test that context can be modified after creation."""
        error = LLaMAFactoryError("Test", context={"initial": "value"})
        error.context["additional"] = "data"
        assert error.context == {"initial": "value", "additional": "data"}
