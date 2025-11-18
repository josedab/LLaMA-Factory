# RFC-0003: Custom Exception Hierarchy

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Implement a custom exception hierarchy to improve error handling, debugging, and programmatic error management throughout the codebase.

---

## Motivation

Current state: All errors use standard Python exceptions (`ValueError`, `RuntimeError`, `OSError`). This creates several problems:

1. **Generic Catch Blocks**: Can't distinguish LLaMA-Factory errors from third-party errors
2. **Poor Error Context**: Difficult to add structured metadata
3. **Inconsistent Messages**: Error message quality varies
4. **Hard to Debug**: No error codes for documentation lookup
5. **Difficult Recovery**: Can't handle specific error types programmatically

Example of current limitation:
```python
try:
    run_training()
except ValueError as e:
    # Is this from LLaMA-Factory or from PyTorch/HuggingFace?
    # Can't tell without parsing the message string
    pass
```

---

## Detailed Design

### Exception Hierarchy

```python
# src/llamafactory/exceptions.py

class LLaMAFactoryError(Exception):
    """Base exception for all LLaMA-Factory errors.

    Attributes:
        message: Human-readable error description
        error_code: Unique error identifier (e.g., "LF001")
        context: Additional structured context
    """

    def __init__(
        self,
        message: str,
        error_code: str = None,
        context: dict = None,
        hint: str = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        if self.hint:
            parts.append(f"\nHint: {self.hint}")
        return " ".join(parts)


# Configuration Errors
class ConfigurationError(LLaMAFactoryError):
    """Errors in configuration/arguments."""
    pass


class InvalidArgumentError(ConfigurationError):
    """Invalid argument value."""
    pass


class IncompatibleArgumentsError(ConfigurationError):
    """Arguments that cannot be used together."""
    pass


class MissingArgumentError(ConfigurationError):
    """Required argument not provided."""
    pass


# Model Errors
class ModelError(LLaMAFactoryError):
    """Errors related to model loading/processing."""
    pass


class ModelLoadingError(ModelError):
    """Failed to load model."""
    pass


class TokenizerError(ModelError):
    """Failed to load or use tokenizer."""
    pass


class AdapterError(ModelError):
    """Error with adapter (LoRA, etc.)."""
    pass


class IncompatibleModelError(ModelError):
    """Model not compatible with requested operation."""
    pass


# Data Errors
class DataError(LLaMAFactoryError):
    """Errors related to data loading/processing."""
    pass


class DatasetNotFoundError(DataError):
    """Dataset file or registry entry not found."""
    pass


class DataFormatError(DataError):
    """Invalid data format."""
    pass


class DataProcessingError(DataError):
    """Error during data processing."""
    pass


# Training Errors
class TrainingError(LLaMAFactoryError):
    """Errors during training."""
    pass


class CheckpointError(TrainingError):
    """Error loading/saving checkpoint."""
    pass


class DistributedTrainingError(TrainingError):
    """Error in distributed training setup."""
    pass


# Inference Errors
class InferenceError(LLaMAFactoryError):
    """Errors during inference."""
    pass


class EngineError(InferenceError):
    """Error with inference engine."""
    pass


class GenerationError(InferenceError):
    """Error during text generation."""
    pass


# Dependency Errors
class DependencyError(LLaMAFactoryError):
    """Missing or incompatible dependency."""
    pass


class MissingDependencyError(DependencyError):
    """Required package not installed."""
    pass


class IncompatibleVersionError(DependencyError):
    """Package version incompatible."""
    pass
```

### Error Codes

Assign unique codes for documentation and debugging:

```python
# src/llamafactory/exceptions.py

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
```

### Usage Examples

#### Before
```python
# src/llamafactory/hparams/parser.py
if model_args.adapter_name_or_path and finetuning_args.finetuning_type != "lora":
    raise ValueError("Adapter is only valid for the LoRA method.")
```

#### After
```python
# src/llamafactory/hparams/parser.py
from llamafactory.exceptions import IncompatibleArgumentsError

if model_args.adapter_name_or_path and finetuning_args.finetuning_type != "lora":
    raise IncompatibleArgumentsError(
        message="Adapter is only valid for the LoRA method",
        error_code="LF101",
        context={
            "adapter_name_or_path": model_args.adapter_name_or_path,
            "finetuning_type": finetuning_args.finetuning_type,
        },
        hint="Either remove adapter_name_or_path or set finetuning_type='lora'"
    )
```

#### Error Handling
```python
from llamafactory.exceptions import (
    LLaMAFactoryError,
    ModelLoadingError,
    DatasetNotFoundError,
)

try:
    run_training(config)
except ModelLoadingError as e:
    logger.error(f"Model issue: {e.message}")
    logger.debug(f"Context: {e.context}")
    # Handle model-specific recovery
except DatasetNotFoundError as e:
    logger.error(f"Dataset issue: {e.message}")
    # Handle dataset-specific recovery
except LLaMAFactoryError as e:
    # Catch-all for LLaMA-Factory errors
    logger.error(f"Training failed: {e}")
except Exception as e:
    # Third-party errors
    logger.error(f"Unexpected error: {e}")
```

---

## Implementation Plan

### Phase 1: Foundation (Days 1-2)
1. Create `src/llamafactory/exceptions.py`
2. Define exception hierarchy
3. Implement error codes dictionary
4. Add to package `__init__.py`

### Phase 2: Configuration Module (Days 3-4)
5. Update `hparams/parser.py` (~30 ValueError → ConfigurationError)
6. Update `hparams/*_args.py` validation
7. Add tests for new exceptions

### Phase 3: Model Module (Days 5-6)
8. Update `model/loader.py`
9. Update `model/adapter.py`
10. Update `model/patcher.py`

### Phase 4: Data Module (Days 7-8)
11. Update `data/loader.py`
12. Update `data/converter.py`
13. Update `data/processor/*.py`

### Phase 5: Remaining Modules (Days 9-10)
14. Update training workflows
15. Update API/chat modules
16. Documentation and cleanup

---

## Backwards Compatibility

### Breaking Changes
Users catching specific standard exceptions will need to update:

```python
# Old code (will still work for now)
try:
    run_training()
except ValueError as e:
    handle_error(e)

# New code (recommended)
try:
    run_training()
except ConfigurationError as e:
    handle_error(e)
```

### Migration Strategy

1. **Phase 1**: Add custom exceptions alongside existing ones
2. **Phase 2**: Deprecation warnings for catching standard exceptions
3. **Phase 3**: Complete migration in next major version

```python
# Transition period - raise both for compatibility
class ConfigurationError(LLaMAFactoryError, ValueError):
    """Inherit from ValueError for backwards compatibility."""
    pass
```

---

## Alternatives Considered

### 1. Error Codes Without Custom Exceptions
- Rejected: Loses the benefit of programmatic handling
- Can't catch by type

### 2. Single Generic Exception
- Rejected: Can't distinguish error types
- Loses granularity benefits

### 3. Exception per Module
- Rejected: Too many exception types
- Current hierarchy provides good balance

---

## Open Questions

1. **Should we include stack traces in context?**
   - Suggested: Only in DEBUG mode

2. **Should errors be localized?**
   - Suggested: Not for v1, but design for it

3. **Should we add retry suggestions?**
   - Suggested: Yes, via the `hint` field

---

## Success Criteria

- [ ] All errors raised are custom exceptions
- [ ] Every exception has error code assigned
- [ ] Context includes relevant variables
- [ ] Hints provide actionable guidance
- [ ] Error codes documented in troubleshooting guide
- [ ] Tests cover all exception types

---

## Effort Estimation

**Total: 8-10 dev-days**

| Task | Effort |
|------|--------|
| Exception hierarchy | 1 day |
| Configuration module | 2 days |
| Model module | 2 days |
| Data module | 2 days |
| Remaining modules | 2 days |
| Documentation | 1 day |

---

## Rollback Strategy

1. Keep ValueError/RuntimeError inheritance during transition
2. If issues arise, users can catch parent exception types
3. Remove inheritance in next major version after validation

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] At least 3 community reviewers (breaking change)
- [ ] Documentation maintainer
