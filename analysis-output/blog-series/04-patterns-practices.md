# Blog 4: Patterns and Practices in LLaMA-Factory

**Analysis Commit SHA:** `45f0437`
**Reading Time:** ~10 minutes

---

## What You'll Learn

- Key design patterns used throughout the codebase
- Configuration management approach
- Error handling and validation patterns
- Testing strategies and quality practices

---

## Introduction

Good architecture is necessary but not sufficient for maintainable software. How code is organized, how errors are handled, and how patterns are applied consistently—these practices determine whether a codebase is pleasant or painful to work with.

Let's examine the patterns and practices that make LLaMA-Factory work.

---

## Design Patterns

### Factory Pattern

The Factory pattern is used extensively for creating objects based on configuration:

```python
# src/llamafactory/train/tuner.py (lines 40-70)
def run_exp(dict_config: dict) -> None:
    # Route to appropriate factory method
    stage = finetuning_args.stage

    if stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif stage == "rm":
        run_rm(...)
    elif stage == "dpo":
        run_dpo(...)
    elif stage == "ppo":
        run_ppo(...)
    elif stage == "kto":
        run_kto(...)
```

[View source: tuner.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/train/tuner.py#L40-L70)

This pattern allows:
- Adding new training stages without modifying existing code
- Each stage having its own dependencies and setup
- Clear routing based on configuration

### Strategy Pattern

Inference engines use the Strategy pattern:

```python
# src/llamafactory/chat/base_engine.py (lines 20-60)
class BaseEngine(ABC):
    """Abstract base for inference engines."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **kwargs,
    ) -> list[str]:
        """Generate responses."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream responses."""
        pass


# src/llamafactory/chat/hf_engine.py
class HuggingFaceEngine(BaseEngine):
    def chat(self, messages, system=None, **kwargs):
        # HuggingFace-specific implementation
        pass


# src/llamafactory/chat/vllm_engine.py
class VllmEngine(BaseEngine):
    def chat(self, messages, system=None, **kwargs):
        # vLLM-specific implementation
        pass
```

[View source: base_engine.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/chat/base_engine.py#L20-L60)

### Template Method Pattern

Dataset converters use Template Method:

```python
# src/llamafactory/data/converter.py (lines 30-70)
class DatasetConverter:
    """Base converter with template method."""

    def __call__(self, examples):
        # Template method
        outputs = self._initialize_outputs()

        for i in range(len(examples[self.primary_key])):
            # Hook methods - override in subclasses
            prompt = self._extract_prompt(examples, i)
            response = self._extract_response(examples, i)
            system = self._extract_system(examples, i)
            medias = self._extract_medias(examples, i)

            outputs["prompt"].append(prompt)
            outputs["response"].append(response)
            outputs["system"].append(system)
            self._append_medias(outputs, medias)

        return outputs

    def _extract_prompt(self, examples, i):
        """Override in subclass."""
        raise NotImplementedError

    def _extract_response(self, examples, i):
        """Override in subclass."""
        raise NotImplementedError
```

[View source: converter.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/data/converter.py#L30-L70)

### Observer Pattern (Callbacks)

Training uses Observer pattern through callbacks:

```python
# src/llamafactory/train/callbacks.py (lines 100-150)
class LogCallback(TrainerCallback):
    """Observer for training events."""

    def on_train_begin(self, args, state, control, **kwargs):
        # Handle training start
        self.start_time = time.time()
        self._create_thread_pool(args.output_dir)

    def on_log(self, args, state, control, **kwargs):
        # Handle logging events
        self._write_metrics(state.log_history[-1])

    def on_train_end(self, args, state, control, **kwargs):
        # Clean up
        self.thread_pool.shutdown(wait=True)

    def on_step_end(self, args, state, control, **kwargs):
        # Check for abort signal
        if self.aborted:
            control.should_training_stop = True
```

[View source: callbacks.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/train/callbacks.py#L100-L150)

### Decorator Pattern (Adapters)

LoRA and other adapters use Decorator pattern:

```python
# src/llamafactory/model/adapter.py (lines 50-100)
def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> "PeftModel":
    """Wrap model with adapter (decorator pattern)."""

    if finetuning_args.finetuning_type == "lora":
        peft_config = LoraConfig(
            r=finetuning_args.lora_rank,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            target_modules=finetuning_args.lora_target,
        )
    elif finetuning_args.finetuning_type == "oft":
        peft_config = OFTConfig(
            r=finetuning_args.oft_rank,
            # ...
        )

    # Wrap model with PEFT adapter
    model = get_peft_model(model, peft_config)

    # Model now has adapter interface but same API
    return model
```

[View source: adapter.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/model/adapter.py#L50-L100)

---

## Configuration Management

### Dataclass-Based Arguments

Configuration uses typed dataclasses with validation:

```python
# src/llamafactory/hparams/model_args.py (lines 30-100)
@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models."
        }
    )

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of bits to quantize the model using bitsandbytes.",
            "choices": [4, 8],
        }
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer."}
    )

    def __post_init__(self):
        """Validation after initialization."""
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.quantization_bit is not None and self.quantization_bit not in [4, 8]:
            raise ValueError("quantization_bit must be 4 or 8")
```

[View source: model_args.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/hparams/model_args.py#L30-L100)

### Configuration Merging

OmegaConf enables configuration composition:

```python
# src/llamafactory/hparams/parser.py (lines 50-80)
def parse_args(parser, args=None):
    """Parse args from CLI, YAML, or JSON."""
    if args is None:
        args = sys.argv[1:]

    # Check if first arg is a config file
    if args and args[0].endswith((".yaml", ".yml", ".json")):
        config_file = args[0]
        remaining_args = args[1:]

        # Load config file
        config = OmegaConf.load(config_file)

        # Apply CLI overrides
        # e.g., llamafactory-cli train config.yaml model_name_or_path=llama3
        for arg in remaining_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                OmegaConf.update(config, key, value)

        # Convert to argument dict
        args_dict = OmegaConf.to_container(config)
        return parser.parse_dict(args_dict)

    return parser.parse_args_into_dataclasses(args)
```

[View source: parser.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/hparams/parser.py#L50-L80)

---

## Error Handling Patterns

### Validation at Boundaries

Input validation happens at system boundaries:

```python
# src/llamafactory/hparams/parser.py (lines 120-180)
def _verify_model_args(model_args, data_args, finetuning_args):
    """Validate argument combinations."""

    # Adapter method compatibility
    if model_args.adapter_name_or_path and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for LoRA method.")

    # Quantization constraints
    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError(
                "Quantization is only compatible with LoRA or OFT method."
            )

        if model_args.adapter_name_or_path and len(model_args.adapter_name_or_path) != 1:
            raise ValueError(
                "Quantized model only accepts a single adapter. Merge them first."
            )

    # Visual model requirements
    if model_args.visual_inputs and not model_args.image_resolution:
        raise ValueError(
            "Please specify `image_resolution` for visual models."
        )
```

[View source: parser.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/hparams/parser.py#L120-L180)

### Graceful Fallbacks

When possible, try alternatives before failing:

```python
# src/llamafactory/model/loader.py (lines 78-100)
def load_tokenizer(model_args):
    """Load tokenizer with fallback."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:
        # Try flipping fast tokenizer setting
        logger.info("Fast tokenizer failed, trying slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        raise OSError(f"Failed to load tokenizer: {e}") from e

    return tokenizer
```

[View source: loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/model/loader.py#L78-L100)

### Actionable Error Messages

Errors include fix suggestions:

```python
# src/llamafactory/extras/misc.py (lines 76-92)
def check_version(requirement: str, mandatory: bool = False) -> None:
    """Check package version with helpful error message."""
    if is_env_enabled("DISABLE_VERSION_CHECK") and not mandatory:
        logger.warning_once("Version checking disabled, may lead to issues.")
        return

    # Construct helpful hint
    if "gptqmodel" in requirement or "autoawq" in requirement:
        pip_command = f"pip install {requirement} --no-build-isolation"
    else:
        pip_command = f"pip install {requirement}"

    if mandatory:
        hint = f"To fix: run `{pip_command}`."
    else:
        hint = f"To fix: run `{pip_command}` or set DISABLE_VERSION_CHECK=1."

    require_version(requirement, hint)
```

[View source: misc.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/extras/misc.py#L76-L92)

---

## Code Organization

### Module Responsibility

Each module has a clear, single responsibility:

| Module | Responsibility |
|--------|---------------|
| `hparams/` | Configuration parsing and validation |
| `model/` | Model loading and modification |
| `data/` | Data loading and processing |
| `train/` | Training orchestration |
| `chat/` | Inference execution |
| `api/` | HTTP API endpoints |
| `webui/` | Web interface |
| `extras/` | Cross-cutting utilities |

### Import Structure

Imports follow a clear hierarchy to avoid cycles:

```
extras (utilities, no internal deps)
   ↑
hparams (config, imports extras)
   ↑
model, data (domain, import hparams, extras)
   ↑
train, chat, eval (services, import domain)
   ↑
api, webui (entry points, import services)
```

### Lazy Imports

Optional dependencies are imported lazily:

```python
# src/llamafactory/train/trainer_utils.py (lines 680-700)
def get_swanlab_callback(finetuning_args):
    """Get SwanLab callback with lazy import."""
    # Only import if actually used
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback

    if finetuning_args.swanlab_api_key:
        swanlab.login(api_key=finetuning_args.swanlab_api_key)

    return SwanLabCallback(
        project=finetuning_args.swanlab_project,
        # ...
    )
```

[View source: trainer_utils.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/train/trainer_utils.py#L680-L700)

---

## Testing Approach

### Test Organization

Tests mirror source structure:

```
tests/
├── data/              # Tests for data module
│   ├── processor/     # Processor tests
│   └── test_*.py
├── model/             # Tests for model module
├── train/             # Tests for train module
├── e2e/               # End-to-end tests
└── conftest.py        # Shared fixtures
```

### Custom Pytest Markers

Tests use markers for selective execution:

```python
# tests/conftest.py (lines 20-60)
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "skip_on_devices: skip on specific devices")
    config.addinivalue_line("markers", "require_device: require specific device")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless RUN_SLOW=1."""
    if not os.getenv("RUN_SLOW"):
        skip_slow = pytest.mark.skip(reason="Set RUN_SLOW=1 to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# Usage in tests
@pytest.mark.slow
def test_full_training():
    """This test is slow and skipped by default."""
    pass


@pytest.mark.skip_on_devices("npu", "xpu")
def test_cuda_specific():
    """Skip on NPU and XPU."""
    pass
```

[View source: conftest.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/tests/conftest.py#L20-L60)

### Parametrized Tests

Tests cover multiple scenarios efficiently:

```python
# tests/data/test_template.py
@pytest.mark.parametrize("template_name", ["llama3", "qwen", "mistral", "chatglm"])
@pytest.mark.parametrize("use_fast", [True, False])
def test_template_encoding(template_name, use_fast):
    """Test template encoding for different models."""
    template = get_template(template_name)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_FOR_TEMPLATE[template_name],
        use_fast=use_fast
    )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    encoded = template.encode_oneturn(tokenizer, messages)
    assert len(encoded) > 0
```

---

## Quality Practices

### Pre-commit Hooks

Code quality enforced before commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: check-ast
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### Ruff Configuration

Consistent code style:

```toml
# pyproject.toml
[tool.ruff]
target-version = "py39"
line-length = 119
indent-width = 4

[tool.ruff.lint]
extend-select = ["C", "E", "F", "I", "W", "UP", "D", "PT009", "RUF022"]
ignore = ["E501", "D100", "D101", "D102", "D103", "D104", "D105", "D107"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

### CI Pipeline

GitHub Actions validates all changes:

```yaml
# .github/workflows/tests.yml
jobs:
  test:
    strategy:
      matrix:
        python: [3.9, 3.10, 3.11, 3.12]
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - name: Code Quality
        run: |
          ruff check .
          ruff format --check .

      - name: License Check
        run: make license

      - name: Tests
        run: pytest -vv tests/
```

---

## Key Takeaways

1. **Consistent Patterns**: Factory, Strategy, Template Method, Observer used throughout
2. **Strong Typing**: Dataclass-based configuration with `__post_init__` validation
3. **Graceful Degradation**: Try alternatives before failing, provide actionable messages
4. **Clear Boundaries**: Validation at entry points, single responsibility per module
5. **Lazy Loading**: Optional dependencies imported only when needed
6. **Automated Quality**: Pre-commit hooks and CI enforce standards

---

## What's Next

In [Blog 5: Extending and Integrating](./05-extending-integrating.md), we'll put these patterns to work by walking through how to add new models, datasets, and training methods to LLaMA-Factory.

---

## References

- [Design Patterns (Gang of Four)](https://en.wikipedia.org/wiki/Design_Patterns)
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Ruff Linter](https://docs.astral.sh/ruff/)
