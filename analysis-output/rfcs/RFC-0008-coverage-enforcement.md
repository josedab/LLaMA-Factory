# RFC-0008: Test Coverage Enforcement

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Implement test coverage tracking and enforcement to ensure code quality, prevent regressions, and identify untested code paths.

---

## Motivation

Currently, LLaMA-Factory has:
- Test structure (32 files, 3,156 LOC)
- No coverage measurement
- No coverage requirements
- No visibility into untested code

This leads to:
1. Unknown code coverage (estimated 60-70%)
2. No protection against coverage regression
3. Uncertainty about test completeness
4. New code may not have tests

Industry standard for ML frameworks: 80%+ coverage.

---

## Detailed Design

### Coverage Configuration

```toml
# pyproject.toml

[tool.coverage.run]
source = ["src/llamafactory"]
branch = true
omit = [
    "*/third_party/*",
    "*/__init__.py",
    "*/v1/*",  # Experimental, separate coverage
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "@abstractmethod",
]
show_missing = true
skip_covered = false

[tool.coverage.html]
directory = "coverage_html"

[tool.coverage.xml]
output = "coverage.xml"
```

### Coverage Targets

#### Phase 1: Baseline (50%)
- Establish current coverage
- No regression allowed

#### Phase 2: Growth (70%)
- Focus on critical paths
- Require coverage for new code

#### Phase 3: Target (80%)
- Industry standard
- Comprehensive coverage

### Per-Module Targets

| Module | Target | Priority |
|--------|--------|----------|
| `hparams/` | 90% | High (configuration validation) |
| `data/` | 85% | High (data correctness) |
| `model/` | 80% | High (model loading) |
| `train/` | 75% | Medium (integration-heavy) |
| `chat/` | 80% | High (user-facing) |
| `api/` | 85% | High (user-facing) |
| `webui/` | 60% | Low (UI testing harder) |
| `extras/` | 75% | Medium |

### CI Integration

```yaml
# .github/workflows/tests.yml

- name: Run Tests with Coverage
  run: |
    pip install pytest-cov
    pytest --cov=src/llamafactory --cov-report=xml --cov-report=html tests/

- name: Check Coverage
  run: |
    coverage report --fail-under=70

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    fail_ci_if_error: true
```

### Coverage Report Integration

Add Codecov/Coveralls badge:
```markdown
[![Coverage](https://codecov.io/gh/hiyouga/LLaMA-Factory/branch/main/graph/badge.svg)](https://codecov.io/gh/hiyouga/LLaMA-Factory)
```

### PR Coverage Check

```yaml
# .github/workflows/pr-coverage.yml

name: PR Coverage Check

on:
  pull_request:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - name: Check Coverage Diff
        uses: orgoro/coverage@v3
        with:
          coverageFile: coverage.xml
          token: ${{ secrets.GITHUB_TOKEN }}
          thresholdAll: 0.7
          thresholdNew: 0.8  # New code must have 80% coverage
```

---

## Example Usage

### Running Coverage Locally

```bash
# Run with coverage
pytest --cov=src/llamafactory --cov-report=term-missing tests/

# Generate HTML report
pytest --cov=src/llamafactory --cov-report=html tests/
open coverage_html/index.html

# Check specific module
pytest --cov=src/llamafactory/data --cov-report=term tests/data/
```

### Adding Tests for Coverage

```python
# tests/data/test_loader.py

import pytest
from llamafactory.data.loader import get_dataset, load_dataset_info

class TestDatasetLoader:
    """Tests for dataset loading functionality."""

    def test_load_dataset_info_from_default(self):
        """Test loading dataset info from default location."""
        info = load_dataset_info()
        assert "alpaca_en" in info

    def test_load_dataset_info_custom_path(self):
        """Test loading from custom path."""
        info = load_dataset_info("data")
        assert isinstance(info, dict)

    def test_load_dataset_info_missing_file(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            load_dataset_info("/nonexistent/path")

    @pytest.mark.parametrize("source", ["hf_hub", "local", "cloud"])
    def test_get_dataset_sources(self, source, mock_dataset):
        """Test loading from different sources."""
        # Test each data source type
        pass
```

### Marking Uncoverable Code

```python
def __repr__(self):  # pragma: no cover
    return f"Model({self.name})"

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol
```

---

## Implementation Plan

### Phase 1: Setup (Week 1)
1. Add coverage configuration
2. CI integration (non-blocking)
3. Generate baseline report
4. Add Codecov integration

### Phase 2: Quick Wins (Weeks 2-3)
5. Add tests for uncovered utilities
6. Test configuration validation
7. Test data format converters
8. Target: 65% coverage

### Phase 3: Core Modules (Weeks 4-5)
9. Model loading tests
10. Data processing tests
11. API endpoint tests
12. Target: 75% coverage

### Phase 4: Enforcement (Week 6)
13. Enable fail-under in CI
14. Enable PR coverage check
15. Target: 80% coverage

---

## Backwards Compatibility

No breaking changes. Coverage is a development tool only.

---

## Alternatives Considered

### 1. Line Coverage Only
- Rejected: Branch coverage catches more bugs

### 2. 100% Coverage Target
- Rejected: Diminishing returns, some code hard to test

### 3. No Enforcement
- Rejected: Coverage will regress without requirements

---

## Open Questions

1. **Should coverage failures block merges?**
   - Suggested: Yes, after reaching 70% baseline

2. **Separate coverage for slow tests?**
   - Suggested: Yes, fast tests in CI, full coverage nightly

3. **Coverage for generated code?**
   - Suggested: Exclude from metrics

---

## Success Criteria

- [ ] Coverage tracking in CI
- [ ] Codecov integration active
- [ ] PR coverage comments enabled
- [ ] 80%+ overall coverage achieved
- [ ] No module below 60%
- [ ] New code requires 80% coverage

---

## Effort Estimation

**Total: 15-20 dev-days**

| Task | Effort |
|------|--------|
| Setup and configuration | 2 days |
| Baseline tests | 5 days |
| Core module tests | 8 days |
| CI integration | 2 days |
| Documentation | 1 day |

---

## Metrics to Track

- Overall coverage percentage
- Coverage by module
- Coverage trend over time
- Uncovered lines count
- Branch coverage percentage

---

## Rollback Strategy

1. Remove coverage requirement from CI
2. Keep coverage tracking (visibility)
3. Re-enable after test improvements

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] QA lead/contributor
- [ ] CI/CD maintainer
