# RFC-0004: Optimizer Builder Pattern

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Extract common optimizer building logic into a shared builder class to eliminate 170 lines of code duplication between GaLore and APOLLO optimizers.

---

## Motivation

Analysis found significant code duplication in `src/llamafactory/train/trainer_utils.py`:

- **Lines 198-284** (GaLore optimizer): 86 lines
- **Lines 286-367** (APOLLO optimizer): 81 lines
- **Overlap**: ~85% similar code (170 duplicated lines)

Both functions:
1. Group parameters by layer type
2. Apply different learning rates per group
3. Handle frozen parameters
4. Configure optimizer-specific settings

This duplication:
- Increases maintenance burden (changes needed in multiple places)
- Risks inconsistencies when fixing bugs
- Makes adding new optimizers harder
- Violates DRY principle

---

## Detailed Design

### Current Structure

```python
# src/llamafactory/train/trainer_utils.py (lines 198-284)
def create_galore_optimizer(model, training_args, finetuning_args):
    # Group parameters
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group = {"params": [param]}

        # Set learning rate by layer type
        if "embed" in name:
            group["lr"] = training_args.learning_rate * 0.1
        elif "lm_head" in name:
            group["lr"] = training_args.learning_rate * 0.1
        else:
            group["lr"] = training_args.learning_rate

        # GaLore-specific settings
        if should_use_galore(name, finetuning_args):
            group["rank"] = finetuning_args.galore_rank
            group["update_proj_gap"] = finetuning_args.galore_update_gap
            group["scale"] = finetuning_args.galore_scale
            group["proj_type"] = finetuning_args.galore_proj_type

        param_groups.append(group)

    return GaLoreAdamW(param_groups, **optimizer_kwargs)


# Lines 286-367 - Nearly identical for APOLLO
def create_apollo_optimizer(model, training_args, finetuning_args):
    # 85% identical code...
```

### Proposed Structure

```python
# src/llamafactory/train/optimizer_builder.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class OptimizerConfig:
    """Configuration for optimizer building."""
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float


class OptimizerBuilder(ABC):
    """Base class for building optimizers with parameter grouping."""

    def __init__(
        self,
        model: "PreTrainedModel",
        config: OptimizerConfig,
        finetuning_args: "FinetuningArguments",
    ):
        self.model = model
        self.config = config
        self.finetuning_args = finetuning_args

    def build(self) -> "Optimizer":
        """Build the optimizer with configured parameter groups."""
        param_groups = self._create_parameter_groups()
        optimizer_kwargs = self._get_optimizer_kwargs()
        return self._create_optimizer(param_groups, optimizer_kwargs)

    def _create_parameter_groups(self) -> list[dict[str, Any]]:
        """Create parameter groups with layer-specific learning rates."""
        param_groups = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            group = {"params": [param], "name": name}

            # Apply layer-specific learning rate
            group["lr"] = self._get_layer_lr(name)

            # Apply weight decay
            group["weight_decay"] = self._get_weight_decay(name)

            # Apply optimizer-specific settings
            self._apply_optimizer_settings(group, name)

            param_groups.append(group)

        return param_groups

    def _get_layer_lr(self, name: str) -> float:
        """Get learning rate for a layer based on its name."""
        if "embed" in name or "lm_head" in name:
            return self.config.learning_rate * 0.1
        return self.config.learning_rate

    def _get_weight_decay(self, name: str) -> float:
        """Get weight decay for a layer."""
        if "bias" in name or "norm" in name:
            return 0.0
        return self.config.weight_decay

    def _get_optimizer_kwargs(self) -> dict[str, Any]:
        """Get common optimizer keyword arguments."""
        return {
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_epsilon,
        }

    @abstractmethod
    def _apply_optimizer_settings(self, group: dict, name: str) -> None:
        """Apply optimizer-specific settings to parameter group."""
        pass

    @abstractmethod
    def _create_optimizer(self, param_groups: list, kwargs: dict) -> "Optimizer":
        """Create the optimizer instance."""
        pass


class GaLoreOptimizerBuilder(OptimizerBuilder):
    """Builder for GaLore optimizer."""

    def _apply_optimizer_settings(self, group: dict, name: str) -> None:
        if self._should_use_galore(name):
            group["rank"] = self.finetuning_args.galore_rank
            group["update_proj_gap"] = self.finetuning_args.galore_update_gap
            group["scale"] = self.finetuning_args.galore_scale
            group["proj_type"] = self.finetuning_args.galore_proj_type

    def _should_use_galore(self, name: str) -> bool:
        # Existing logic for determining GaLore applicability
        return any(target in name for target in self.finetuning_args.galore_target)

    def _create_optimizer(self, param_groups: list, kwargs: dict) -> "Optimizer":
        from galore_torch import GaLoreAdamW
        return GaLoreAdamW(param_groups, **kwargs)


class APOLLOOptimizerBuilder(OptimizerBuilder):
    """Builder for APOLLO optimizer."""

    def _apply_optimizer_settings(self, group: dict, name: str) -> None:
        if self._should_use_apollo(name):
            group["rank"] = self.finetuning_args.apollo_rank
            group["scale"] = self.finetuning_args.apollo_scale
            group["warmup"] = self.finetuning_args.apollo_warmup

    def _should_use_apollo(self, name: str) -> bool:
        return any(target in name for target in self.finetuning_args.apollo_target)

    def _create_optimizer(self, param_groups: list, kwargs: dict) -> "Optimizer":
        from apollo_torch import APOLLOAdamW
        return APOLLOAdamW(param_groups, **kwargs)


# Factory function for backwards compatibility
def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "Optimizer":
    """Create optimizer based on configuration."""
    config = OptimizerConfig(
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        adam_beta1=training_args.adam_beta1,
        adam_beta2=training_args.adam_beta2,
        adam_epsilon=training_args.adam_epsilon,
    )

    if finetuning_args.use_galore:
        builder = GaLoreOptimizerBuilder(model, config, finetuning_args)
    elif finetuning_args.use_apollo:
        builder = APOLLOOptimizerBuilder(model, config, finetuning_args)
    else:
        raise ValueError("No custom optimizer requested")

    return builder.build()
```

### Benefits

1. **170 lines → ~100 lines** (40% reduction)
2. **Single point of change** for parameter grouping logic
3. **Easy to add new optimizers** (just implement 2 abstract methods)
4. **Testable** - can unit test base class behavior
5. **Consistent** - all optimizers use same grouping logic

---

## Example Usage

### Adding a New Optimizer

```python
class AdamMiniOptimizerBuilder(OptimizerBuilder):
    """Builder for Adam-mini optimizer."""

    def _apply_optimizer_settings(self, group: dict, name: str) -> None:
        # Adam-mini specific settings
        group["compress_ratio"] = self.finetuning_args.adam_mini_ratio

    def _create_optimizer(self, param_groups: list, kwargs: dict) -> "Optimizer":
        from adam_mini import AdamMini
        return AdamMini(param_groups, **kwargs)
```

---

## Implementation Plan

### Phase 1: Create Builder (Days 1-2)
1. Create `optimizer_builder.py`
2. Implement base class and tests
3. Create GaLore builder

### Phase 2: Migrate APOLLO (Day 3)
4. Create APOLLO builder
5. Update trainer_utils.py imports
6. Run tests

### Phase 3: Cleanup (Days 4-5)
7. Remove old functions
8. Update documentation
9. Add example for new optimizer creation

---

## Backwards Compatibility

### API Changes
The external API remains unchanged:
```python
# Still works
trainer = Trainer(
    model=model,
    args=TrainingArguments(use_galore=True, ...),
)
```

### Internal Changes
- Old functions marked deprecated
- New builder used internally
- No user-facing changes

---

## Alternatives Considered

### 1. Simple Function Extraction
```python
def create_param_groups(model, training_args):
    # Shared logic
    pass

def create_galore_optimizer(model, training_args, finetuning_args):
    groups = create_param_groups(model, training_args)
    # Add GaLore-specific settings
```
- Rejected: Still requires duplication for optimizer-specific settings

### 2. Configuration-Based Builder
- Rejected: Less type-safe, harder to extend

---

## Open Questions

1. **Should BAdam use the same builder?**
   - Yes, add BAdam builder for consistency

2. **Should we support custom parameter grouping rules?**
   - Future enhancement via callback

---

## Success Criteria

- [ ] Code duplication reduced to < 2%
- [ ] All optimizer tests pass
- [ ] New optimizer can be added with < 50 LOC
- [ ] No performance regression

---

## Effort Estimation

**Total: 4-5 dev-days**

| Task | Effort |
|------|--------|
| Base builder implementation | 1.5 days |
| GaLore/APOLLO builders | 1 day |
| Testing | 1 day |
| Documentation | 0.5 day |

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] Optimizer contributors
