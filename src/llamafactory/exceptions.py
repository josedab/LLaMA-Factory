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

r"""
Custom exception hierarchy for LLaMA-Factory.

This module provides a structured exception hierarchy to improve error handling,
debugging, and programmatic error management throughout the codebase.

Example usage:
    >>> from llamafactory.exceptions import ConfigurationError, InvalidArgumentError
    >>> raise InvalidArgumentError(
    ...     message="Invalid batch size value",
    ...     error_code="LF100",
    ...     context={"batch_size": -1},
    ...     hint="Batch size must be a positive integer"
    ... )
"""

from typing import Any, Optional


# Error codes for documentation and debugging
ERROR_CODES = {
    # Configuration (LF1xx)
    "LF100": "Invalid argument value",
    "LF101": "Incompatible arguments",
    "LF102": "Missing required argument",
    "LF103": "Unknown argument",
    # Model (LF2xx)
    "LF200": "Model not found",
    "LF201": "Model loading failed",
    "LF202": "Tokenizer loading failed",
    "LF203": "Adapter initialization failed",
    "LF204": "Incompatible model architecture",
    # Data (LF3xx)
    "LF300": "Dataset not found",
    "LF301": "Invalid data format",
    "LF302": "Data processing failed",
    "LF303": "Empty dataset",
    # Training (LF4xx)
    "LF400": "Training initialization failed",
    "LF401": "Checkpoint loading failed",
    "LF402": "Checkpoint saving failed",
    "LF403": "Distributed setup failed",
    # Inference (LF5xx)
    "LF500": "Engine initialization failed",
    "LF501": "Generation failed",
    "LF502": "Invalid request",
    # Dependencies (LF6xx)
    "LF600": "Missing dependency",
    "LF601": "Incompatible version",
}


class LLaMAFactoryError(Exception):
    """Base exception for all LLaMA-Factory errors.

    This is the base class for all custom exceptions in LLaMA-Factory.
    It provides structured error information including error codes,
    context data, and hints for resolution.

    Attributes:
        message: Human-readable error description.
        error_code: Unique error identifier (e.g., "LF001").
        context: Additional structured context about the error.
        hint: Actionable suggestion for resolving the error.

    Example:
        >>> raise LLaMAFactoryError(
        ...     message="Something went wrong",
        ...     error_code="LF001",
        ...     context={"key": "value"},
        ...     hint="Try doing X instead"
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        hint: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message with code and hint."""
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        if self.hint:
            parts.append(f"\nHint: {self.hint}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"context={self.context!r}, "
            f"hint={self.hint!r})"
        )


# ============================================================================
# Configuration Errors (LF1xx)
# ============================================================================


class ConfigurationError(LLaMAFactoryError, ValueError):
    """Errors in configuration/arguments.

    Inherits from ValueError for backwards compatibility with existing
    code that catches ValueError for configuration issues.
    """

    pass


class InvalidArgumentError(ConfigurationError):
    """Invalid argument value provided.

    Raised when an argument has an invalid value (wrong type, out of range, etc.).
    """

    pass


class IncompatibleArgumentsError(ConfigurationError):
    """Arguments that cannot be used together.

    Raised when multiple arguments are provided that are mutually exclusive
    or incompatible with each other.
    """

    pass


class MissingArgumentError(ConfigurationError):
    """Required argument not provided.

    Raised when a required argument is missing from the configuration.
    """

    pass


# ============================================================================
# Model Errors (LF2xx)
# ============================================================================


class ModelError(LLaMAFactoryError, RuntimeError):
    """Errors related to model loading/processing.

    Inherits from RuntimeError for backwards compatibility.
    """

    pass


class ModelLoadingError(ModelError):
    """Failed to load model.

    Raised when a model cannot be loaded from disk or hub.
    """

    pass


class TokenizerError(ModelError):
    """Failed to load or use tokenizer.

    Raised when tokenizer loading or processing fails.
    """

    pass


class AdapterError(ModelError):
    """Error with adapter (LoRA, etc.).

    Raised when there's an issue with adapter initialization, loading, or merging.
    """

    pass


class IncompatibleModelError(ModelError):
    """Model not compatible with requested operation.

    Raised when a model architecture doesn't support the requested operation.
    """

    pass


# ============================================================================
# Data Errors (LF3xx)
# ============================================================================


class DataError(LLaMAFactoryError, ValueError):
    """Errors related to data loading/processing.

    Inherits from ValueError for backwards compatibility.
    """

    pass


class DatasetNotFoundError(DataError, FileNotFoundError):
    """Dataset file or registry entry not found.

    Raised when a specified dataset cannot be found in the registry or filesystem.
    """

    pass


class DataFormatError(DataError):
    """Invalid data format.

    Raised when data doesn't match the expected format or schema.
    """

    pass


class DataProcessingError(DataError):
    """Error during data processing.

    Raised when data processing (tokenization, conversion, etc.) fails.
    """

    pass


# ============================================================================
# Training Errors (LF4xx)
# ============================================================================


class TrainingError(LLaMAFactoryError, RuntimeError):
    """Errors during training.

    Inherits from RuntimeError for backwards compatibility.
    """

    pass


class CheckpointError(TrainingError):
    """Error loading/saving checkpoint.

    Raised when checkpoint operations fail.
    """

    pass


class DistributedTrainingError(TrainingError):
    """Error in distributed training setup.

    Raised when distributed training initialization or communication fails.
    """

    pass


# ============================================================================
# Inference Errors (LF5xx)
# ============================================================================


class InferenceError(LLaMAFactoryError, RuntimeError):
    """Errors during inference.

    Inherits from RuntimeError for backwards compatibility.
    """

    pass


class EngineError(InferenceError):
    """Error with inference engine.

    Raised when inference engine initialization or operation fails.
    """

    pass


class GenerationError(InferenceError):
    """Error during text generation.

    Raised when text generation fails due to model or configuration issues.
    """

    pass


# ============================================================================
# Dependency Errors (LF6xx)
# ============================================================================


class DependencyError(LLaMAFactoryError, ImportError):
    """Missing or incompatible dependency.

    Inherits from ImportError for backwards compatibility with
    import-related error handling.
    """

    pass


class MissingDependencyError(DependencyError):
    """Required package not installed.

    Raised when a required optional dependency is not available.
    """

    pass


class IncompatibleVersionError(DependencyError):
    """Package version incompatible.

    Raised when an installed package version doesn't meet requirements.
    """

    pass


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Error codes
    "ERROR_CODES",
    # Base exception
    "LLaMAFactoryError",
    # Configuration errors
    "ConfigurationError",
    "InvalidArgumentError",
    "IncompatibleArgumentsError",
    "MissingArgumentError",
    # Model errors
    "ModelError",
    "ModelLoadingError",
    "TokenizerError",
    "AdapterError",
    "IncompatibleModelError",
    # Data errors
    "DataError",
    "DatasetNotFoundError",
    "DataFormatError",
    "DataProcessingError",
    # Training errors
    "TrainingError",
    "CheckpointError",
    "DistributedTrainingError",
    # Inference errors
    "InferenceError",
    "EngineError",
    "GenerationError",
    # Dependency errors
    "DependencyError",
    "MissingDependencyError",
    "IncompatibleVersionError",
]
