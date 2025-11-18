# Contributing to LLaMA-Factory

Thank you for your interest in contributing to LLaMA-Factory! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code Style](#code-style)
- [Documentation Standards](#documentation-standards)
  - [Module Docstrings](#module-docstrings)
  - [Function and Class Docstrings](#function-and-class-docstrings)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The configuration is in `pyproject.toml`.

Key style requirements:
- Python 3.9+ syntax
- Line length: 119 characters
- 4-space indentation
- Double quotes for strings
- Google-style docstrings

Run the linter before submitting:

```bash
ruff check src/llamafactory
ruff format src/llamafactory
```

## Documentation Standards

All Python modules must have comprehensive docstrings following the Google style guide.

### Module Docstrings

Every Python file must have a module-level docstring immediately after the license header and before any imports.

#### Template

```python
# Copyright 2025 the LlamaFactory team.
# ... license header ...

"""Short one-line description of the module.

Longer description that explains:
- What this module does
- Key abstractions it provides
- How it fits in the overall architecture
- When to use/not use it

Key Functions:
    function_name: Brief description

Key Classes:
    ClassName: Brief description

Example:
    Basic usage example::

        from llamafactory.module import function
        result = function(...)

Attributes:
    MODULE_CONSTANT: Description of any module-level constants

Note:
    Any important caveats or warnings

See Also:
    - Related modules
    - External documentation
"""

import ...
```

#### Requirements

Each module docstring must include:
- [ ] One-line summary (imperative mood, e.g., "Provide..." not "Provides...")
- [ ] Longer description (what and why)
- [ ] Key functions/classes list
- [ ] At least one usage example
- [ ] See Also references to related modules

#### Examples by Module Type

**Entry Point Modules:**

```python
"""Command-line interface entry point for LLaMA-Factory.

This module provides the main entry point for the llamafactory-cli command.
It checks for USE_V1 environment variable to determine which launcher to use.

Key Functions:
    main: Main entry point function

Example:
    From command line::

        llamafactory-cli train config.yaml
        llamafactory-cli api --model_name_or_path llama3

See Also:
    - launcher.py for command routing and distributed training setup
"""
```

**Core Domain Modules:**

```python
"""Dataset loading and preprocessing for LLaMA-Factory.

This module handles loading datasets from multiple sources (HuggingFace Hub,
ModelScope, cloud storage, local files) and preparing them for training.

Key Functions:
    get_dataset: Main entry point for dataset loading
    load_dataset_info: Load dataset registry

Data Flow:
    Source -> Loader -> Converter -> Processor -> Collator

Example:
    Load and process a dataset::

        from llamafactory.data.loader import get_dataset

        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            tokenizer=tokenizer,
        )

See Also:
    - converter.py for format conversion
    - processor/ for stage-specific processing
"""
```

**Utility Modules:**

```python
"""Custom logging system for LLaMA-Factory.

Provides a logging framework with support for:
- Distributed training (rank-aware logging)
- Async file logging for WebUI
- Configurable verbosity levels

Key Functions:
    get_logger: Get a logger instance
    info_rank0: Log only from rank 0 process
    warning_rank0_once: Warn once per message from rank 0

Environment Variables:
    LLAMAFACTORY_VERBOSITY: Set logging level (DEBUG, INFO, WARNING, ERROR)

Example:
    Get and use a logger::

        from llamafactory.extras.logging import get_logger

        logger = get_logger(__name__)
        logger.info_rank0("Training started")

Note:
    In distributed training, use rank0 variants to avoid duplicate logs.

See Also:
    - extras.constants for logging-related constants
"""
```

### Function and Class Docstrings

Follow Google-style docstrings for functions and classes:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Short description of what the function does.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this error is raised.

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass


class ClassName:
    """Short description of the class.

    Longer description if needed.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.

    Example:
        >>> obj = ClassName()
        >>> obj.method()
    """

    def __init__(self, attr1: str):
        """Initialize ClassName.

        Args:
            attr1: Description of attr1.
        """
        self.attr1 = attr1
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Testing

Run tests before submitting:

```bash
pytest tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes following the style guidelines
4. Run linting and tests
5. Submit a pull request

### Commit Message Format

Use clear, descriptive commit messages:

```
[component] Brief description of changes

Longer description if needed, explaining:
- What was changed
- Why it was changed
- Any breaking changes
```

Example:
```
[data] Add support for ShareGPT format

- Implement ShareGPT converter
- Update dataset registry
- Add tests for new format
```

## Questions?

If you have questions, please open an issue on GitHub.
