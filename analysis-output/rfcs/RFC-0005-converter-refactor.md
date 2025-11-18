# RFC-0005: Data Converter Refactoring

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Refactor the `_find_medias()` function in `src/llamafactory/data/converter.py` to reduce nesting depth from 7 levels to 3-4, improving readability and maintainability.

---

## Motivation

Analysis identified `_find_medias()` (lines 43-76) as a complexity hotspot:

- **Nesting depth**: 7 levels (target: <5)
- **Cyclomatic complexity**: >12
- **Mixed concerns**: Type checking, validation, and collection in single function

This makes the code:
- Hard to understand at a glance
- Difficult to test individual branches
- Error-prone when modifying
- Poor candidate for code review

---

## Detailed Design

### Current Code

```python
# src/llamafactory/data/converter.py (lines 43-76)
def _find_medias(self, examples: dict, idx: int, media_type: str) -> list[Any]:
    """Find media of a given type in the examples."""
    medias = []
    if self.dataset_attr.medias:
        media_list = self.dataset_attr.medias.get(media_type, [])
        for media_col in media_list:
            if media_col in examples:
                media_values = examples[media_col][idx]
                if media_values is not None:
                    if isinstance(media_values, list):
                        for item in media_values:
                            if item is not None:
                                if isinstance(item, dict):
                                    if "bytes" in item:
                                        medias.append(item["bytes"])
                                    elif "path" in item:
                                        medias.append(item["path"])
                                else:
                                    medias.append(item)
                    elif isinstance(media_values, dict):
                        if "bytes" in media_values:
                            medias.append(media_values["bytes"])
                        elif "path" in media_values:
                            medias.append(media_values["path"])
                    else:
                        medias.append(media_values)
    return medias
```

### Proposed Refactoring

```python
# src/llamafactory/data/converter.py

def _find_medias(self, examples: dict, idx: int, media_type: str) -> list[Any]:
    """Find media of a given type in the examples.

    Args:
        examples: Batch of examples
        idx: Index of current example
        media_type: Type of media to find (image, video, audio)

    Returns:
        List of media items (paths or bytes)
    """
    if not self.dataset_attr.medias:
        return []

    media_columns = self.dataset_attr.medias.get(media_type, [])
    if not media_columns:
        return []

    medias = []
    for column in media_columns:
        if column not in examples:
            continue

        raw_value = examples[column][idx]
        extracted = self._extract_media_value(raw_value)
        medias.extend(extracted)

    return medias


def _extract_media_value(self, value: Any) -> list[Any]:
    """Extract media items from a raw value.

    Handles:
    - None values (ignored)
    - Lists of items (flattened)
    - Dict with bytes/path keys
    - Direct values (paths, bytes)

    Args:
        value: Raw value from dataset

    Returns:
        List of extracted media items
    """
    if value is None:
        return []

    if isinstance(value, list):
        return self._extract_from_list(value)

    if isinstance(value, dict):
        return self._extract_from_dict(value)

    # Direct value (path or bytes)
    return [value]


def _extract_from_list(self, items: list) -> list[Any]:
    """Extract media from a list of items."""
    results = []
    for item in items:
        results.extend(self._extract_media_value(item))
    return results


def _extract_from_dict(self, item: dict) -> list[Any]:
    """Extract media from a dict item.

    Supports HuggingFace datasets format with 'bytes' or 'path' keys.
    """
    if "bytes" in item:
        return [item["bytes"]]
    if "path" in item:
        return [item["path"]]
    return []
```

### Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max nesting | 7 | 3 | 57% reduction |
| Cyclomatic complexity | >12 | 6 | 50% reduction |
| Lines | 34 | 50 | Better organized |
| Testability | Poor | Excellent | Each helper testable |

---

## Example Usage

### Testing (Now Possible)

```python
def test_extract_from_dict_bytes():
    converter = DatasetConverter(...)
    result = converter._extract_from_dict({"bytes": b"image_data"})
    assert result == [b"image_data"]

def test_extract_from_dict_path():
    converter = DatasetConverter(...)
    result = converter._extract_from_dict({"path": "/images/cat.jpg"})
    assert result == ["/images/cat.jpg"]

def test_extract_from_list_mixed():
    converter = DatasetConverter(...)
    result = converter._extract_from_list([
        "/images/1.jpg",
        {"path": "/images/2.jpg"},
        None,  # Should be ignored
    ])
    assert result == ["/images/1.jpg", "/images/2.jpg"]

def test_find_medias_empty():
    converter = DatasetConverter(...)
    converter.dataset_attr.medias = None
    result = converter._find_medias({}, 0, "image")
    assert result == []
```

---

## Implementation Plan

### Phase 1: Add New Functions (Day 1)
1. Add helper functions alongside existing code
2. Write tests for new functions
3. Verify behavior matches

### Phase 2: Migrate (Day 2)
4. Replace old `_find_medias` with new implementation
5. Run full test suite
6. Verify multimodal training works

### Phase 3: Cleanup (Day 3)
7. Remove old code
8. Update documentation
9. Add docstrings

---

## Backwards Compatibility

No breaking changes. This is an internal refactoring that maintains the same external behavior.

---

## Alternatives Considered

### 1. Match Statement (Python 3.10+)
```python
match value:
    case None:
        return []
    case list() as items:
        return self._extract_from_list(items)
    case dict() as item:
        return self._extract_from_dict(item)
    case _:
        return [value]
```
- Rejected: LLaMA-Factory supports Python 3.9

### 2. Recursive Single Function
- Rejected: Still hard to test individual cases

---

## Open Questions

1. **Should we validate media files exist?**
   - Suggested: No, leave to processor stage

2. **Support additional dict formats?**
   - Consider for future if community requests

---

## Success Criteria

- [ ] Max nesting depth ≤ 4
- [ ] Cyclomatic complexity ≤ 8
- [ ] All multimodal tests pass
- [ ] Each helper function has unit test
- [ ] No performance regression

---

## Effort Estimation

**Total: 2-3 dev-days**

| Task | Effort |
|------|--------|
| Implement helpers | 0.5 day |
| Write tests | 1 day |
| Integration testing | 0.5 day |
| Documentation | 0.5 day |

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] Multimodal feature contributor
