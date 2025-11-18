# RFC-0007: Constants Module Refactoring

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Split the 3,762-line `constants.py` file into domain-specific modules to improve maintainability, navigation, and import performance.

---

## Motivation

`src/llamafactory/extras/constants.py` has grown to 3,762 lines, containing:

- Model definitions (2,000+ lines)
- File extension mappings
- Default values
- Template names
- Training stage mappings
- Miscellaneous constants

Problems:
1. **Hard to navigate**: Finding specific constants requires extensive scrolling
2. **Import overhead**: Loading all constants when only some are needed
3. **Merge conflicts**: Multiple contributors editing same file
4. **Testing difficulty**: Can't easily mock specific constant groups
5. **Single file with multiple concerns**: Violates separation of concerns

---

## Detailed Design

### Current Structure

```
src/llamafactory/extras/
├── constants.py  (3,762 LOC - everything)
└── ...
```

### Proposed Structure

```
src/llamafactory/extras/
├── constants/
│   ├── __init__.py        # Re-exports for compatibility
│   ├── models.py          # SUPPORTED_MODELS, model-related
│   ├── templates.py       # TEMPLATES, DEFAULT_TEMPLATE
│   ├── training.py        # Training stages, methods
│   ├── data.py            # File types, data formats
│   └── defaults.py        # Default values, paths
└── ...
```

### File Contents

#### models.py (~2,000 LOC)
```python
"""Model definitions and support mappings.

This module contains the SUPPORTED_MODELS dictionary and
related model architecture mappings.
"""

SUPPORTED_MODELS: dict[str, dict] = {
    "llama": {
        "LlamaForCausalLM": {
            "model_type": "llama",
            # ...
        }
    },
    "qwen": {
        # ...
    },
    # ... all model definitions
}

MODEL_ARCHITECTURES = {
    "LlamaForCausalLM": "llama",
    "QwenForCausalLM": "qwen",
    # ...
}

def get_model_info(model_name: str) -> dict:
    """Get model information by name."""
    return SUPPORTED_MODELS.get(model_name, {})
```

#### templates.py (~200 LOC)
```python
"""Template-related constants."""

TEMPLATES: dict[str, str] = {
    "llama3": "llama3",
    "qwen": "qwen",
    # ...
}

DEFAULT_TEMPLATE = "default"

SYSTEM_PROMPTS: dict[str, str] = {
    "default": "You are a helpful assistant.",
    # ...
}
```

#### training.py (~300 LOC)
```python
"""Training-related constants."""

from enum import Enum

class Stage(str, Enum):
    PT = "pt"
    SFT = "sft"
    RM = "rm"
    DPO = "dpo"
    PPO = "ppo"
    KTO = "kto"

class FinetuningType(str, Enum):
    FULL = "full"
    FREEZE = "freeze"
    LORA = "lora"
    OFT = "oft"

STAGE_TO_PROCESSOR = {
    Stage.PT: "pretrain",
    Stage.SFT: "supervised",
    Stage.RM: "pairwise",
    Stage.DPO: "pairwise",
    Stage.PPO: "supervised",
    Stage.KTO: "pairwise",
}
```

#### data.py (~200 LOC)
```python
"""Data-related constants."""

FILEEXT2TYPE = {
    "json": "json",
    "jsonl": "json",
    "csv": "csv",
    "parquet": "parquet",
    "arrow": "arrow",
}

IGNORE_INDEX = -100

DATA_SOURCES = ["hf_hub_url", "ms_hub_url", "cloud_path", "file_name"]
```

#### defaults.py (~100 LOC)
```python
"""Default values and paths."""

DEFAULT_DATA_DIR = "data"
DEFAULT_CACHE_DIR = "cache"

CHECKPOINT_NAMES = [
    "adapter_model.bin",
    "adapter_model.safetensors",
    "pytorch_model.bin",
    "model.safetensors",
]

LOG_FILE_NAMES = {
    "trainer_log": "trainer_log.jsonl",
    "running_log": "running_log.txt",
}
```

#### __init__.py (Compatibility Layer)
```python
"""Constants module - backwards compatible imports.

Import from submodules for new code:
    from llamafactory.extras.constants.models import SUPPORTED_MODELS

Or use compatibility imports:
    from llamafactory.extras.constants import SUPPORTED_MODELS
"""

# Re-export everything for backwards compatibility
from .models import SUPPORTED_MODELS, MODEL_ARCHITECTURES, get_model_info
from .templates import TEMPLATES, DEFAULT_TEMPLATE, SYSTEM_PROMPTS
from .training import Stage, FinetuningType, STAGE_TO_PROCESSOR
from .data import FILEEXT2TYPE, IGNORE_INDEX, DATA_SOURCES
from .defaults import (
    DEFAULT_DATA_DIR,
    DEFAULT_CACHE_DIR,
    CHECKPOINT_NAMES,
    LOG_FILE_NAMES,
)

__all__ = [
    # Models
    "SUPPORTED_MODELS",
    "MODEL_ARCHITECTURES",
    "get_model_info",
    # Templates
    "TEMPLATES",
    "DEFAULT_TEMPLATE",
    "SYSTEM_PROMPTS",
    # Training
    "Stage",
    "FinetuningType",
    "STAGE_TO_PROCESSOR",
    # Data
    "FILEEXT2TYPE",
    "IGNORE_INDEX",
    "DATA_SOURCES",
    # Defaults
    "DEFAULT_DATA_DIR",
    "DEFAULT_CACHE_DIR",
    "CHECKPOINT_NAMES",
    "LOG_FILE_NAMES",
]
```

---

## Example Usage

### Current (Still Works)
```python
from llamafactory.extras.constants import SUPPORTED_MODELS, IGNORE_INDEX
```

### New (Recommended)
```python
from llamafactory.extras.constants.models import SUPPORTED_MODELS
from llamafactory.extras.constants.data import IGNORE_INDEX
```

### Selective Import
```python
# Only loads model definitions, not everything
from llamafactory.extras.constants.models import SUPPORTED_MODELS
```

---

## Implementation Plan

### Phase 1: Create Structure (Day 1-2)
1. Create `constants/` directory
2. Extract models.py (largest)
3. Extract templates.py
4. Create `__init__.py` with re-exports

### Phase 2: Continue Extraction (Day 3-4)
5. Extract training.py
6. Extract data.py
7. Extract defaults.py

### Phase 3: Update Imports (Day 5-6)
8. Find all imports using grep
9. Optionally update to new paths
10. Verify all tests pass

### Phase 4: Documentation (Day 7)
11. Update contributing guide
12. Add deprecation notes for old import paths (optional)

---

## Backwards Compatibility

### Fully Compatible
All existing imports continue to work:
```python
from llamafactory.extras.constants import SUPPORTED_MODELS  # Works
```

### Optional Migration
New imports recommended but not required:
```python
from llamafactory.extras.constants.models import SUPPORTED_MODELS  # Preferred
```

---

## Alternatives Considered

### 1. Move to JSON/YAML Files
```yaml
# models.yaml
llama:
  LlamaForCausalLM:
    model_type: llama
```
- Rejected: Loses type checking, IDE support
- Constants with logic can't be in data files

### 2. Lazy Loading
```python
_models = None
def get_supported_models():
    global _models
    if _models is None:
        _models = load_models()
    return _models
```
- Rejected: More complex, less IDE-friendly

### 3. Single Large File with Sections
- Rejected: Doesn't solve core issues

---

## Open Questions

1. **Should old import paths emit deprecation warnings?**
   - Suggested: No for v1, consider for v2

2. **Should we use Enums more widely?**
   - Suggested: Yes, as shown in training.py

3. **Version the constants module?**
   - Suggested: Not necessary, track in main version

---

## Success Criteria

- [ ] No file > 1,000 LOC
- [ ] All existing imports work
- [ ] Tests pass without modification
- [ ] Each submodule has clear purpose
- [ ] Documentation updated

---

## Effort Estimation

**Total: 6-8 dev-days**

| Task | Effort |
|------|--------|
| Create structure | 1 day |
| Extract modules | 3 days |
| Update imports | 2 days |
| Testing | 1 day |
| Documentation | 0.5 day |

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] Community feedback (may affect plugins)
