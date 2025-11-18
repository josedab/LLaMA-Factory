# Metrics Summary

**Analysis Commit SHA:** `45f0437`
**Analysis Date:** 2025-11-18

---

## Code Size Metrics

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | 30,783 |
| **Python Files** | 125 |
| **Test Files** | 32 |
| **Average File Size** | 246 LOC |
| **Largest File** | `constants.py` (3,762 LOC) |

### Module Breakdown

| Module | Files | LOC | Avg LOC/File | Primary Responsibility |
|--------|-------|-----|--------------|----------------------|
| **data** | 17 | 7,067 | 416 | Data processing & loading |
| **webui** | 19 | 5,975 | 314 | Web interface |
| **train** | 32 | 4,661 | 146 | Training workflows |
| **extras** | 7 | 4,576 | 654 | Utilities & constants |
| **model** | 21 | 3,350 | 160 | Model loading & optimization |
| **hparams** | 7 | 2,800+ | 400 | Configuration |
| **chat** | 7 | 1,575 | 225 | Inference engines |
| **api** | 5 | 677 | 135 | REST API |
| **eval** | 3 | 237 | 79 | Evaluation |

### File Size Distribution

| Size Range | Count | Percentage |
|------------|-------|------------|
| < 100 LOC | 35 | 28% |
| 100-300 LOC | 58 | 46% |
| 300-500 LOC | 18 | 14% |
| 500-1000 LOC | 10 | 8% |
| > 1000 LOC | 4 | 3% |

### Largest Files (Potential Refactoring Candidates)

1. `extras/constants.py` - 3,762 LOC (model definitions)
2. `webui/locales.py` - 3,178 LOC (i18n strings)
3. `data/template.py` - ~800 LOC (chat templates)
4. `train/callbacks.py` - ~600 LOC (training callbacks)

---

## Code Quality Metrics

### Complexity Analysis

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Max Nesting Depth** | 7 levels | < 5 | ⚠️ Needs Work |
| **High Complexity Functions** | 3 | 0 | ⚠️ Needs Work |
| **Functions with >5 Parameters** | ~2% | < 5% | ✅ Good |
| **Average Function Length** | ~25 LOC | < 30 | ✅ Good |

### Complexity Hotspots

1. **`data/converter.py:_find_medias()`** - Lines 43-76
   - Nesting: 7 levels
   - Cyclomatic Complexity: >12

2. **`data/converter.py:AlpacaDatasetConverter.__call__()`** - Lines 86-131
   - Length: 138 lines
   - Branches: 8+

3. **`webui/runner.py:_parse_train_args()`** - Lines 120-185
   - Length: 166 lines
   - Nesting: 4 levels

### Code Duplication

| Area | Duplicated LOC | Impact |
|------|----------------|--------|
| GaLore/APOLLO optimizers | 170 | High |
| ToolUtils classes (8 similar) | 100-150 | Medium |
| Dataset converters | ~80 | Low |
| **Total Estimated** | **5-7%** | Target: <2% |

---

## Documentation Metrics

### Docstring Coverage

| Scope | Coverage | Target | Status |
|-------|----------|--------|--------|
| **Functions** | 52% (326/625) | 75% | ⚠️ Below Target |
| **Files with Docstrings** | 50% (62/125) | 80% | ⚠️ Below Target |
| **Module-level Docstrings** | 0.8% (1/125) | 100% | ❌ Critical Gap |

### Documentation by Module

| Module | Coverage | Assessment |
|--------|----------|------------|
| data | 59% | Best |
| model | 57% | Good |
| chat | 57% | Good |
| webui | 53% | Acceptable |
| train | 47% | Below Target |
| **api** | **40%** | ❌ Should be 100% |
| eval | 33% | Poor |

### External Documentation

- README.md: Present (English)
- README_zh.md: Present (Chinese)
- Wiki: Available on GitHub
- API Documentation: Generated from docstrings (incomplete)
- Examples: 50+ configuration files

---

## Test Metrics

### Test Coverage Structure

| Test Type | Files | Description |
|-----------|-------|-------------|
| Unit Tests | ~15 | Data, model, formatting |
| Integration Tests | ~8 | Pipeline, loading |
| End-to-End Tests | 3 | Training, chat, SGLang |
| License Checks | 1 | Header validation |
| **Total** | **32** | |

### Test Code Statistics

| Metric | Value |
|--------|-------|
| Test LOC (main) | 2,882 |
| Test LOC (v1) | 274 |
| **Total Test LOC** | **3,156** |
| Test/Code Ratio | ~10% |

### CI/CD Pipeline

| Platform | Python Versions | OS Platforms |
|----------|-----------------|--------------|
| GitHub Actions | 3.9, 3.10, 3.11, 3.12 | Ubuntu, Windows, macOS |

### Coverage Configuration

- **Explicit Coverage Targets**: Not configured
- **Coverage Gate**: Not enforced
- **Estimated Coverage**: ~60-70% (based on test structure)

---

## Import Structure

### Module Dependencies

| Module | Incoming Imports | Risk Level |
|--------|-----------------|------------|
| extras | 79 | Low (utilities) |
| hparams | 49 | Low (config) |
| data | 22 | Medium |
| model | 17 | Medium |
| trainer_utils | 13 | Low |

### Import Health

- **Circular Import Risk**: None detected ✅
- **Maximum Import Chain**: 4 levels ✅
- **TYPE_CHECKING Usage**: Properly applied ✅

---

## Dependency Metrics

### Core Dependencies

| Category | Count | Key Packages |
|----------|-------|--------------|
| Core ML | 5 | transformers, datasets, accelerate, peft, trl |
| Web/API | 3 | fastapi, gradio, uvicorn |
| Data Processing | 3 | numpy, pandas, scipy |
| Tokenization | 3 | sentencepiece, tiktoken, modelscope |
| Config | 3 | pyyaml, omegaconf, pydantic |
| **Total Core** | **30** | |

### Optional Extras

| Category | Count | Examples |
|----------|-------|----------|
| Hardware Acceleration | 5 | deepspeed, liger-kernel, bitsandbytes |
| Quantization | 4 | gptq, aqlm, hqq, eetq |
| Inference Engines | 2 | vllm, sglang |
| Optimizers | 4 | galore, apollo, badam, adam-mini |
| **Total Extras** | **23 groups** | |

### Version Pinning Strategy

| Strategy | Count | Examples |
|----------|-------|----------|
| Range Constraints | 12 | `transformers>=4.49.0,<=4.57.1` |
| Min Version Only | 8 | `modelscope>=1.14.0` |
| Max Version Only | 5 | `numpy<2.0.0` |
| Exact Pins | 3 | `torch==2.7.1` (NPU) |
| Unpinned | 6 | `fire`, `scipy` |

---

## Performance Metrics

### Supported Features

| Feature | Options |
|---------|---------|
| Training Stages | 6 (PT, SFT, RM, DPO, PPO, KTO) |
| Adapter Methods | 6 (LoRA, QLoRA, OFT, DoRA, PiSSA, LLaMA Pro) |
| Quantization Methods | 8+ (BNB, GPTQ, AWQ, AQLM, EETQ, HQQ, Quanto, MXFP4) |
| Inference Engines | 4 (HF, vLLM, SGLang, KTransformers) |
| Hardware Backends | 3 (CUDA, ROCm, NPU) |

### Optimization Integrations

- FlashAttention-2
- Unsloth
- Liger Kernel
- RoPE Scaling
- Gradient Checkpointing
- Mixed Precision (FP16, BF16, FP8)

---

## Security Metrics

### Critical Issues Found

| Issue | Severity | Location |
|-------|----------|----------|
| API verbose logging default ON | High | `api/chat.py:83-84` |
| CORS misconfiguration | High | `api/app.py:72-78` |
| Missing request timeouts | High | `api/chat.py` |
| Extra args validation bypass | Medium | `webui/runner.py:179` |
| No WebUI authentication | High | `webui/interface.py` |

### Security Positives

- SSRF protection: ✅ Implemented
- LFI protection: ✅ Implemented
- Safe subprocess usage: ✅ No shell=True
- Safe YAML: ✅ Uses safe_load()
- No eval/exec: ✅ Clean

---

## Quality Scorecard

| Metric | Grade | Score |
|--------|-------|-------|
| LOC Organization | B+ | Good |
| Nesting Depth | C | 7/10 |
| Cyclomatic Complexity | B | 7/10 |
| Parameter Complexity | A | 9/10 |
| Code Duplication | C | 5/10 |
| Function Documentation | C+ | 6/10 |
| Module Documentation | F | 1/10 |
| Import Structure | A+ | 10/10 |
| Test Structure | B | 7/10 |
| Security | C | 5/10 |
| **OVERALL** | **B-** | **6.7/10** |

---

## Improvement Priorities

### Quick Wins (< 1 week)
1. Add module docstrings to all files (~2 days)
2. Fix 5 security vulnerabilities (~1 day)
3. Add request timeouts (~2 hours)

### Strategic (2-4 weeks)
1. Extract optimizer builder pattern (saves 170 LOC)
2. Reduce converter nesting (7 → 3-4 levels)
3. Improve API documentation (40% → 90%)
4. Split constants.py into domain modules

### Long-term (> 1 month)
1. Add custom exception hierarchy
2. Implement code coverage enforcement (target: 80%)
3. Create comprehensive integration test suite
4. Implement type checking (mypy)
