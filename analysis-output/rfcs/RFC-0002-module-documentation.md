# RFC-0002: Module-Level Documentation

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Add comprehensive module-level docstrings to all 125 Python files in the codebase, improving developer onboarding, IDE support, and code maintainability.

---

## Motivation

Analysis revealed that only 1 of 125 Python files (0.8%) has a module-level docstring. This creates several problems:

1. **Poor Discoverability**: Developers can't quickly understand module purposes
2. **Weak IDE Support**: No module documentation in autocomplete/hover
3. **Onboarding Friction**: New contributors struggle to navigate codebase
4. **Documentation Gaps**: Auto-generated docs are empty at module level

Module docstrings are the first line of documentation developers see and set the context for all code that follows.

---

## Detailed Design

### Docstring Standard

Follow Google-style docstrings as configured in `pyproject.toml`:

```python
"""Short one-line description of the module.

Longer description that explains:
- What this module does
- Key abstractions it provides
- How it fits in the overall architecture
- When to use/not use it

Example:
    Basic usage example::

        from llamafactory.data import loader
        dataset = loader.get_dataset(...)

Attributes:
    MODULE_CONSTANT: Description of any module-level constants

Note:
    Any important caveats or warnings

See Also:
    - Related modules
    - External documentation
"""
```

### Module Categories

Different modules need different documentation focus:

#### Entry Point Modules
```python
# src/llamafactory/cli.py
"""Command-line interface for LLaMA-Factory.

This module provides the main entry point for the llamafactory-cli command.
It parses the command and routes to the appropriate handler.

Commands:
    train: Launch distributed training
    api: Start OpenAI-compatible API server
    chat: Interactive chat interface
    webui: Launch LLaMA Board training UI
    export: Merge adapters and export model
    env: Print environment information
    version: Show version

Example:
    From command line::

        llamafactory-cli train config.yaml
        llamafactory-cli api --model_name_or_path llama3

See Also:
    - launcher.py for distributed training setup
    - train/tuner.py for training orchestration
"""
```

#### Core Domain Modules
```python
# src/llamafactory/data/loader.py
"""Dataset loading and preprocessing for LLaMA-Factory.

This module handles loading datasets from multiple sources (HuggingFace Hub,
ModelScope, cloud storage, local files) and preparing them for training.

Key Functions:
    get_dataset: Main entry point for dataset loading
    load_dataset_info: Load dataset registry

Data Flow:
    Source → Loader → Converter → Processor → Collator

Supported Sources:
    - HuggingFace Hub (hf_hub_url)
    - ModelScope (ms_hub_url)
    - Cloud storage (S3, GCS via cloud_path)
    - Local files (file_name)

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
    - data/dataset_info.json for dataset registry
"""
```

#### Utility Modules
```python
# src/llamafactory/extras/logging.py
"""Custom logging system for LLaMA-Factory.

Provides a logging framework with support for:
- Distributed training (rank-aware logging)
- Async file logging for WebUI
- Configurable verbosity levels

The logging format is: [LEVEL|timestamp] name:lineno >> message

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
        logger.warning_rank0_once("This warning appears only once")

Note:
    In distributed training, use rank0 variants to avoid duplicate logs.
"""
```

### Priority Order

1. **High Priority** (Week 1):
   - Entry points: `cli.py`, `launcher.py`
   - Core modules: `train/tuner.py`, `model/loader.py`, `data/loader.py`
   - Public API: `api/`, `chat/`

2. **Medium Priority** (Week 2):
   - Domain modules: `hparams/`, `data/`, `model/`
   - Training workflows: `train/*/workflow.py`

3. **Lower Priority** (Week 3):
   - Utilities: `extras/`
   - WebUI: `webui/`
   - Test utilities

### Documentation Requirements

Each docstring must include:
- [ ] One-line summary (imperative mood)
- [ ] Longer description (what and why)
- [ ] Key functions/classes list
- [ ] At least one usage example
- [ ] See Also references

---

## Example Usage

### Before
```python
# src/llamafactory/train/tuner.py
from typing import Any

def run_exp(dict_config: dict[str, Any]) -> None:
    ...
```

### After
```python
# src/llamafactory/train/tuner.py
"""Training orchestration for LLaMA-Factory.

This module is the main entry point for all training operations. It parses
configuration, sets up callbacks, and routes to the appropriate training
workflow based on the stage (PT, SFT, RM, DPO, PPO, KTO).

Key Functions:
    run_exp: Main training entry point
    export_model: Merge adapters and export model

Training Stages:
    - pt: Pre-training (continued training on domain data)
    - sft: Supervised fine-tuning (instruction tuning)
    - rm: Reward model training
    - dpo: Direct preference optimization
    - ppo: Proximal policy optimization (RLHF)
    - kto: Kahneman-Tversky optimization

Example:
    Run training from Python::

        from llamafactory.train.tuner import run_exp

        run_exp({
            "model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "dataset": "alpaca_en",
            "output_dir": "./output",
            "stage": "sft",
        })

    Export a trained model::

        from llamafactory.train.tuner import export_model

        export_model({
            "model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "adapter_name_or_path": "./output",
            "export_dir": "./merged_model",
        })

See Also:
    - launcher.py for distributed training setup
    - train/sft/workflow.py for SFT implementation
    - hparams/ for configuration options
"""
from typing import Any


def run_exp(dict_config: dict[str, Any]) -> None:
    ...
```

---

## Implementation Plan

### Phase 1: Template Creation (Day 1)
1. Create docstring templates for each module category
2. Set up pre-commit hook for docstring validation
3. Document style guide in CONTRIBUTING.md

### Phase 2: High Priority Modules (Days 2-3)
4. Add docstrings to entry points (5 files)
5. Add docstrings to public API (12 files)
6. Add docstrings to core domain (15 files)

### Phase 3: Medium Priority Modules (Days 4-5)
7. Add docstrings to remaining modules (~80 files)
8. Cross-link related modules
9. Validate examples run correctly

### Phase 4: Validation (Day 6)
10. Run documentation generator
11. Review generated documentation
12. Fix any broken links or examples

---

## Backwards Compatibility

No breaking changes. This RFC only adds documentation.

---

## Alternatives Considered

### 1. README Files Per Directory
- Rejected: Separate from code, easily outdated
- Docstrings stay with the code they document

### 2. External Documentation Wiki
- Rejected: Hard to keep synchronized
- Module docstrings generate API docs automatically

### 3. Type Stubs with Documentation
- Rejected: Duplicates information
- Keep documentation with implementation

---

## Open Questions

1. **Should we generate API documentation automatically?**
   - Suggested: Yes, use Sphinx with autodoc

2. **How do we enforce docstring requirements?**
   - Suggested: Add `pydocstyle` or `ruff D` checks to CI

3. **Should we include performance notes?**
   - Suggested: Yes, for modules with significant performance implications

---

## Success Criteria

- [ ] 100% of Python files have module-level docstrings
- [ ] All docstrings follow Google style guide
- [ ] Every docstring includes at least one example
- [ ] Generated API documentation is complete
- [ ] CI check enforces docstring presence

---

## Effort Estimation

**Total: 4-6 dev-days**

| Task | Effort |
|------|--------|
| Template creation | 0.5 day |
| High priority modules (32 files) | 2 days |
| Remaining modules (~93 files) | 2-3 days |
| Validation and review | 0.5 day |

**Velocity**: ~20-25 modules per day with good templates

---

## Rollback Strategy

If this causes issues (unlikely for documentation-only changes):
1. Revert the PR
2. Fix issues in docstrings
3. Re-apply

---

## Stakeholder Approvals

- [ ] Project maintainers (style approval)
- [ ] Documentation maintainer
- [ ] At least 2 community reviewers
