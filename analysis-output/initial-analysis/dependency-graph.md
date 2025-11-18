# Dependency Graph

**Analysis Commit SHA:** `45f0437`

---

## Internal Module Dependencies

### Dependency Flow (Top to Bottom)

```mermaid
graph TD
    CLI[cli.py / launcher.py] --> HPARAMS[hparams]
    CLI --> TRAIN[train]
    CLI --> CHAT[chat]
    CLI --> API[api]
    CLI --> WEBUI[webui]
    CLI --> EVAL[eval]

    TRAIN --> MODEL[model]
    TRAIN --> DATA[data]
    TRAIN --> EXTRAS[extras]

    CHAT --> MODEL
    CHAT --> DATA
    CHAT --> EXTRAS

    API --> CHAT
    API --> EXTRAS

    WEBUI --> TRAIN
    WEBUI --> CHAT
    WEBUI --> EXTRAS

    EVAL --> MODEL
    EVAL --> DATA

    MODEL --> EXTRAS
    DATA --> EXTRAS
    HPARAMS --> EXTRAS

    style CLI fill:#e1f5fe
    style EXTRAS fill:#fff3e0
    style TRAIN fill:#f3e5f5
    style MODEL fill:#e8f5e9
    style DATA fill:#fce4ec
```

### Layer Architecture

```
┌─────────────────────────────────────────────────┐
│              Entry Points (Layer 1)             │
│         cli.py, launcher.py, api/app.py         │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│            Application Layer (Layer 2)          │
│     train/, chat/, webui/, eval/, api/          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Domain Layer (Layer 3)             │
│           model/, data/, hparams/               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│           Infrastructure Layer (Layer 4)        │
│                    extras/                      │
└─────────────────────────────────────────────────┘
```

---

## External Dependencies

### Core ML Framework Dependencies

```mermaid
graph LR
    LF[LLaMA-Factory] --> PT[PyTorch ≥2.0]
    LF --> HF[Transformers 4.49-4.57]
    LF --> DS[Datasets 2.16-4.0]
    LF --> PEFT[PEFT 0.14-0.17]
    LF --> TRL[TRL 0.8-0.9]
    LF --> ACC[Accelerate 1.3-1.11]

    HF --> PT
    PEFT --> HF
    PEFT --> PT
    TRL --> HF
    TRL --> PEFT
    ACC --> PT

    style LF fill:#4caf50,color:white
    style PT fill:#ff9800
    style HF fill:#2196f3
```

### Web & API Dependencies

```mermaid
graph LR
    LF[LLaMA-Factory] --> FA[FastAPI]
    LF --> UV[Uvicorn]
    LF --> GR[Gradio 4.38-5.45]
    LF --> SSE[SSE-Starlette]

    FA --> UV
    GR --> FA

    style LF fill:#4caf50,color:white
```

### Data Processing Dependencies

```mermaid
graph LR
    LF[LLaMA-Factory] --> NP[NumPy <2.0]
    LF --> PD[Pandas ≥2.0]
    LF --> SP[SciPy]
    LF --> ST[SentencePiece]
    LF --> TK[Tiktoken]

    PD --> NP
    SP --> NP

    style LF fill:#4caf50,color:white
```

---

## Optional Dependencies (Extras)

### Hardware Acceleration

```mermaid
graph TD
    subgraph "Hardware Extras"
        CUDA[torch] --> PT[PyTorch ≥2.0]
        NPU[torch-npu] --> PT27[PyTorch 2.7.1]
        DS[deepspeed] --> PT
        LK[liger-kernel]
        BNB[bitsandbytes]
    end

    style CUDA fill:#76ff03
    style NPU fill:#ffd600
```

### Inference Engines

```mermaid
graph LR
    subgraph "Inference Backends"
        VLLM[vLLM 0.4-0.11]
        SG[SGLang ≥0.4.5]
        KT[KTransformers]
    end

    VLLM -.-> |conflicts| SG

    style VLLM fill:#ff5722
    style SG fill:#9c27b0
```

### Quantization Methods

```mermaid
graph TD
    subgraph "Quantization Extras"
        BNB[bitsandbytes]
        GPTQ[gptqmodel ≥2.0]
        AQLM[aqlm ≥1.1]
        HQQ[hqq]
        EETQ[eetq]
    end
```

### Optimizers

```mermaid
graph TD
    subgraph "Optimizer Extras"
        GAL[galore-torch]
        APL[apollo-torch]
        BAD[badam ≥1.2]
        AM[adam-mini]
    end
```

---

## Dependency Conflicts

The following combinations are explicitly prohibited:

```
torch-npu ❌ aqlm
torch-npu ❌ vllm
torch-npu ❌ sglang
vllm ❌ sglang
```

These conflicts are defined in `pyproject.toml` under `[tool.uv.conflicts]`.

---

## Import Relationships

### Most Imported Modules

| Module | Import Count | Role |
|--------|-------------|------|
| `extras` | 79 | Utilities, logging, constants |
| `hparams` | 49 | Configuration dataclasses |
| `data` | 22 | Dataset handling |
| `model` | 17 | Model loading |
| `trainer_utils` | 13 | Training utilities |

### Import Chain Analysis

**Maximum Chain Depth**: 4 levels

```
cli → train → model → extras
cli → webui → train → model
cli → api → chat → model
```

**Circular Import Risk**: None detected ✅

---

## Version Compatibility Matrix

### Python Version Support

| Python | Status | Notes |
|--------|--------|-------|
| 3.9 | ✅ Supported | Some macOS issues |
| 3.10 | ✅ Supported | Primary target |
| 3.11 | ✅ Supported | Good performance |
| 3.12 | ✅ Supported | Latest features |

### Transformers Version Support

| Transformers | Python 3.9 | Python 3.10+ |
|-------------|------------|--------------|
| 4.49.0 | ✅ | ✅ |
| 4.51.0 | ✅ | ✅ |
| 4.53.0 | ✅ | ✅ |
| 4.57.1 | ❌ | ✅ |

Note: `transformers>=4.49.0,<=4.56.2` for Python < 3.10

---

## Dependency Update Strategy

### Version Pinning Philosophy

1. **Core ML (Tight Ranges)**
   - Reason: Breaking API changes common
   - Example: `transformers>=4.49.0,<=4.57.1`

2. **Utilities (Loose/None)**
   - Reason: Stable APIs
   - Example: `fire`, `scipy`

3. **Hardware-Specific (Exact)**
   - Reason: Binary compatibility
   - Example: `torch==2.7.1` (NPU)

### Security Considerations

- Excluded versions with known issues:
  - `transformers!=4.52.0,!=4.57.0`
  - `propcache!=0.4.0`
- NumPy capped at `<2.0.0` for API stability

---

## Recommended Dependency Audit

### Check for:

1. **Abandoned packages**: None detected
2. **Security vulnerabilities**: Run `pip-audit` regularly
3. **License compatibility**: All Apache/MIT/BSD compatible
4. **Outdated packages**: Update quarterly

### Suggested Commands

```bash
# Security audit
pip-audit

# Check for updates
pip list --outdated

# License check
pip-licenses --format=markdown
```
