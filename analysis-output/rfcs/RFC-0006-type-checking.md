# RFC-0006: Type Checking with Mypy

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Implement static type checking using mypy to catch bugs early, improve IDE support, and enforce type consistency across the codebase.

---

## Motivation

Currently, LLaMA-Factory has type hints in many places but no enforcement:

1. **No type validation**: Hints are documentation only
2. **Inconsistent usage**: Some modules well-typed, others not
3. **Runtime errors**: Type bugs caught only at runtime
4. **Poor refactoring safety**: IDE can't catch type mismatches

Benefits of mypy:
- Catch type errors before runtime
- Better IDE autocomplete and error detection
- Documentation that can't drift from code
- Safer refactoring

---

## Detailed Design

### Configuration

```toml
# pyproject.toml

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start permissive, tighten later
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
namespace_packages = true
explicit_package_bases = true

[[tool.mypy.overrides]]
module = "llamafactory.*"
disallow_untyped_defs = true  # Strict for our code
disallow_incomplete_defs = true

[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "datasets.*",
    "peft.*",
    "trl.*",
    "gradio.*",
    "vllm.*",
]
ignore_missing_imports = true
```

### Type Stub Files

For internal modules with complex types:

```python
# src/llamafactory/hparams/_types.pyi

from typing import TypeVar, Union

ModelArgsType = TypeVar("ModelArgsType", bound="ModelArguments")
DataArgsType = TypeVar("DataArgsType", bound="DataArguments")

TrainArgsTuple = tuple[
    "ModelArguments",
    "DataArguments",
    "TrainingArguments",
    "FinetuningArguments",
    "GeneratingArguments",
]
```

### Common Type Issues to Fix

#### 1. Optional Parameters

```python
# Before
def load_model(tokenizer, model_args, finetuning_args, is_trainable):
    ...

# After
def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = True,
) -> "PreTrainedModel":
    ...
```

#### 2. Union Types

```python
# Before
def get_dataset(args):  # args could be dict, list, or None
    ...

# After
def get_dataset(
    args: Optional[Union[dict[str, Any], list[str]]] = None
) -> dict[str, Any]:
    ...
```

#### 3. Callback Types

```python
# Before
callbacks = []

# After
from transformers import TrainerCallback
callbacks: list[TrainerCallback] = []
```

#### 4. Generic Container Types

```python
# Before
def process(examples):
    outputs = {"input_ids": [], "labels": []}
    ...

# After
def process(
    examples: dict[str, list[Any]]
) -> dict[str, list[list[int]]]:
    outputs: dict[str, list[list[int]]] = {
        "input_ids": [],
        "labels": []
    }
    ...
```

### Incremental Adoption Strategy

```
Phase 1: Core infrastructure (extras/, hparams/)
    ↓
Phase 2: Domain layer (model/, data/)
    ↓
Phase 3: Application layer (train/, chat/)
    ↓
Phase 4: Entry points (api/, webui/)
```

### CI Integration

```yaml
# .github/workflows/tests.yml

- name: Type Check
  run: |
    pip install mypy
    mypy src/llamafactory --config-file pyproject.toml
```

---

## Example Usage

### Type-Safe Function

```python
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

def load_model(
    tokenizer: PreTrainedTokenizer,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = True,
) -> PreTrainedModel:
    """Load model with type safety."""
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **get_model_kwargs(model_args),
    )

    if finetuning_args.finetuning_type == "lora":
        model = init_adapter(model, model_args, finetuning_args)

    return model
```

### Protocol for Duck Typing

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for inference engines."""

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        ...

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        ...
```

---

## Implementation Plan

### Phase 1: Setup (Week 1)
1. Add mypy configuration
2. Create type stubs for complex types
3. CI integration (non-blocking initially)

### Phase 2: Infrastructure (Week 2)
4. Type `extras/` module fully
5. Type `hparams/` module fully
6. Fix discovered issues

### Phase 3: Domain (Weeks 3-4)
7. Type `model/` module
8. Type `data/` module
9. Fix discovered issues

### Phase 4: Application (Weeks 5-6)
10. Type `train/` module
11. Type `chat/` module
12. Type `api/` and `webui/`

### Phase 5: Enforcement (Week 7)
13. Enable strict mode
14. Make CI check blocking
15. Document type conventions

---

## Backwards Compatibility

No breaking changes for users. Type hints are:
- Ignored at runtime
- Only checked by mypy during development/CI

---

## Alternatives Considered

### 1. Pyright Instead of Mypy
- Rejected: Mypy more widely used, better HuggingFace support

### 2. Type Hints Without Checker
- Rejected: Hints drift from reality without enforcement

### 3. Runtime Type Checking (pydantic everywhere)
- Rejected: Performance overhead, already using dataclasses

---

## Open Questions

1. **Should we use `from __future__ import annotations`?**
   - Suggested: Yes, for cleaner forward references

2. **How strict should we be?**
   - Start with `disallow_untyped_defs = true` for new code
   - Gradually enable for existing code

3. **Should we type test files?**
   - Suggested: Yes, but with relaxed rules

---

## Success Criteria

- [ ] Zero mypy errors in CI
- [ ] All public functions have type hints
- [ ] All return types specified
- [ ] Type stubs for complex internal types
- [ ] Documentation on type conventions

---

## Effort Estimation

**Total: 25-30 dev-days**

| Task | Effort |
|------|--------|
| Setup and configuration | 2 days |
| Infrastructure modules | 3 days |
| Domain modules | 8 days |
| Application modules | 8 days |
| Testing and refinement | 4 days |

---

## Rollback Strategy

1. Remove mypy from CI requirements
2. Keep type hints (they're still useful as documentation)
3. Re-enable when issues are resolved

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] At least 5 community reviewers (large change)
