# Repository Structure

**Analysis Commit SHA:** `45f0437`

---

## Root Directory

```
/home/user/LLaMA-Factory/
├── src/                          # Main source code
│   └── llamafactory/             # Core package
├── tests/                        # Test suite (pytest)
├── tests_v1/                     # V1 API tests
├── examples/                     # Training configuration examples
├── scripts/                      # Utility scripts
├── docker/                       # Container configurations
├── data/                         # Sample datasets & configs
├── assets/                       # Logos and images
├── .github/                      # CI/CD workflows
│
├── pyproject.toml               # Build & tool configuration
├── setup.py                     # Package metadata & extras
├── requirements.txt             # Core dependencies
├── Makefile                     # Development commands
├── README.md                    # English documentation
└── README_zh.md                 # Chinese documentation
```

---

## Source Code Structure (`src/llamafactory/`)

### Entry Points

| File | Purpose | Key Function |
|------|---------|--------------|
| `cli.py` | Main CLI entry | `main()` |
| `launcher.py` | Command routing & distributed setup | `launch()`, `run_exp()` |

### Core Modules

#### **API** (`api/`) - 677 LOC, 5 files
OpenAI-compatible REST API server.

```
api/
├── __init__.py
├── app.py              # FastAPI application setup
├── chat.py             # Chat completion endpoints
├── common.py           # Security utilities (SSRF/LFI checks)
└── protocol.py         # Request/response models
```

#### **Chat** (`chat/`) - 1,575 LOC, 7 files
Inference engines and chat interface.

```
chat/
├── __init__.py
├── chat_model.py       # Main ChatModel class
├── base_engine.py      # Abstract engine interface
├── hf_engine.py        # HuggingFace Transformers engine
├── vllm_engine.py      # vLLM high-throughput engine
├── sglang_engine.py    # SGLang engine
└── kt_engine.py        # KTransformers CPU engine
```

#### **Data** (`data/`) - 7,067 LOC, 17 files
Data loading, processing, and formatting.

```
data/
├── __init__.py
├── loader.py           # Dataset loading from multiple sources
├── converter.py        # Format conversion (Alpaca, ShareGPT, etc.)
├── template.py         # Chat template system
├── formatter.py        # Data formatting utilities
├── parser.py           # Configuration parsing
├── collator.py         # Batch collation & padding
├── mm_plugin.py        # Multimodal plugin system
├── tool_utils.py       # Tool/function calling utilities
├── data_utils.py       # General data utilities
└── processor/          # Dataset processors
    ├── __init__.py
    ├── processor_utils.py   # Base processor class
    ├── supervised.py        # SFT data processing
    ├── unsupervised.py      # Unsupervised/PT data
    ├── pairwise.py          # DPO/RM pairwise data
    ├── pretrain.py          # Pre-training data
    └── feedback.py          # Feedback/preference data
```

#### **Model** (`model/`) - 3,350 LOC, 21 files
Model loading, adapters, and optimizations.

```
model/
├── __init__.py
├── loader.py           # Model & tokenizer loading
├── adapter.py          # Adapter initialization (LoRA, QLoRA, OFT)
├── patcher.py          # Model patching & customization
└── model_utils/        # Optimization implementations
    ├── attention.py    # Attention optimizations
    ├── longlora.py     # LongLoRA implementation
    ├── unsloth.py      # Unsloth integration
    ├── ktransformers.py# KTransformers support
    ├── liger_kernel.py # Liger kernel optimization
    ├── rms_norm.py     # RMSNorm optimization
    ├── rope.py         # RoPE scaling
    ├── valuehead.py    # Value head for RM/PPO
    └── [13 more files] # Various optimizations
```

#### **Train** (`train/`) - 4,661 LOC, 32 files
Training workflows for all stages.

```
train/
├── __init__.py
├── tuner.py            # Main orchestrator
├── callbacks.py        # Training callbacks
├── trainer_utils.py    # Trainer utilities
├── fp8_utils.py        # FP8 precision utilities
├── test_utils.py       # Testing utilities
│
├── pt/                 # Pre-Training
│   └── workflow.py
├── sft/                # Supervised Fine-Tuning
│   └── workflow.py
├── rm/                 # Reward Modeling
│   └── workflow.py
├── dpo/                # Direct Preference Optimization
│   └── workflow.py
├── ppo/                # Proximal Policy Optimization
│   └── workflow.py
├── kto/                # Kahneman-Tversky Optimization
│   └── workflow.py
├── ksft/               # KTransformers SFT
│   └── workflow.py
└── mca/                # Megatron Core Adapter
    └── workflow.py
```

#### **Hyperparameters** (`hparams/`) - 2,800+ LOC, 7 files
Argument parsing and configuration.

```
hparams/
├── __init__.py
├── parser.py           # Main argument parser
├── model_args.py       # Model configuration
├── data_args.py        # Data configuration
├── training_args.py    # Training configuration
├── finetuning_args.py  # Fine-tuning specific args
├── generating_args.py  # Generation parameters
└── evaluation_args.py  # Evaluation configuration
```

#### **Evaluation** (`eval/`) - 237 LOC, 3 files
Model evaluation capabilities.

```
eval/
├── __init__.py
├── evaluator.py        # Evaluation orchestration
└── template.py         # Evaluation templates
```

#### **Web UI** (`webui/`) - 5,975 LOC, 19 files
Gradio-based training interface (LLaMA Board).

```
webui/
├── __init__.py
├── interface.py        # Main UI interface
├── manager.py          # Resource manager
├── engine.py           # Training engine interface
├── runner.py           # Job runner
├── control.py          # UI controls
├── chatter.py          # Chat interface
├── common.py           # Shared utilities
├── css.py              # Styling
├── locales.py          # Internationalization (3,178 LOC!)
└── components/         # UI components
    └── [various]
```

#### **Extras** (`extras/`) - 4,576 LOC, 7 files
Utilities, logging, and constants.

```
extras/
├── __init__.py
├── constants.py        # Supported models, etc. (3,762 LOC)
├── env.py              # Environment printing
├── logging.py          # Custom logging system
├── misc.py             # Miscellaneous utilities
├── packages.py         # Package version checking
└── ploting.py          # Visualization utilities
```

#### **V1 API** (`v1/`) - Experimental plugin architecture
Alternative modular API with plugin system.

```
v1/
├── launcher.py         # V1 launcher
├── config/             # Configuration system
├── core/               # Core abstractions
├── plugins/            # Plugin system
└── trainers/           # Trainer implementations
```

#### **Third Party** (`third_party/`)
External optimizations.

```
third_party/
└── muon/               # Muon optimizer
```

---

## Test Structure (`tests/`)

```
tests/
├── conftest.py         # Pytest configuration
├── check_license.py    # License validation
│
├── data/               # Data processing tests
│   ├── processor/      # Processor-specific tests
│   └── test_*.py       # Various data tests
├── model/              # Model tests
│   ├── model_utils/    # Utility tests
│   └── test_*.py       # Model loading tests
├── train/              # Training tests
├── eval/               # Evaluation tests
└── e2e/                # End-to-end tests
    ├── test_train.py
    ├── test_chat.py
    └── test_sglang.py
```

---

## Examples Structure (`examples/`)

```
examples/
├── train_lora/         # LoRA training configs
├── train_qlora/        # QLoRA training configs
├── train_full/         # Full fine-tuning configs
├── inference/          # Inference configs
└── extras/             # Advanced configs (GaLore, LoRA+, etc.)
```

---

## Docker Structure (`docker/`)

```
docker/
├── docker-cuda/        # NVIDIA GPU (CUDA)
│   └── Dockerfile
├── docker-rocm/        # AMD GPU (ROCm)
│   └── Dockerfile
└── docker-npu/         # Huawei NPU (Ascend)
    └── Dockerfile
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build system, ruff config, uv conflicts |
| `setup.py` | Package metadata, extras dependencies |
| `requirements.txt` | Core runtime dependencies |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `Makefile` | Development commands |
| `data/dataset_info.json` | Dataset registry |

---

## GitHub Actions (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `tests.yml` | Main CI/CD (lint, test, build) |
| `docker.yml` | Docker image building |
| `publish.yml` | PyPI publishing |
| `label_issue.yml` | Auto-label NPU issues |

---

## Key File Locations for Common Tasks

### Adding a New Model
1. Add to `src/llamafactory/extras/constants.py` → `SUPPORTED_MODELS`
2. Add template in `src/llamafactory/data/template.py` if needed

### Adding a New Training Stage
1. Create `src/llamafactory/train/<stage>/workflow.py`
2. Add stage enum in `src/llamafactory/hparams/finetuning_args.py`
3. Update `src/llamafactory/train/tuner.py` routing

### Adding a New Inference Engine
1. Create `src/llamafactory/chat/<engine>_engine.py`
2. Implement `BaseEngine` interface
3. Add to engine selection in `src/llamafactory/chat/chat_model.py`

### Adding a New Dataset Format
1. Create converter in `src/llamafactory/data/converter.py`
2. Register in format dispatcher

### Adding a New Optimization
1. Create file in `src/llamafactory/model/model_utils/`
2. Integrate in `src/llamafactory/model/patcher.py`
3. Add config in `src/llamafactory/hparams/model_args.py`
