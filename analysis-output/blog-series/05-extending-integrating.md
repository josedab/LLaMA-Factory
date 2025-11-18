# Blog 5: Extending and Integrating LLaMA-Factory

**Analysis Commit SHA:** `45f0437`
**Reading Time:** ~11 minutes

---

## What You'll Learn

- How to add support for new models
- Creating custom datasets
- Implementing new training stages
- API integration patterns
- Common extension scenarios

---

## Introduction

Understanding architecture is valuable, but at some point you need to make changes. Whether you're adding a new model, creating custom datasets, or integrating with your infrastructure, this guide will show you how.

We'll walk through the most common extension scenarios with concrete code examples.

---

## Adding a New Model

Let's add support for a hypothetical "NewModel" with its own tokenizer quirks and chat template.

### Step 1: Add to SUPPORTED_MODELS

```python
# src/llamafactory/extras/constants.py (add after line 500)
SUPPORTED_MODELS["newmodel"] = {
    "NewModelForCausalLM": {
        "model_type": "newmodel",
        "architectures": ["NewModelForCausalLM"],
    }
}
```

[View constants.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/extras/constants.py)

### Step 2: Create Chat Template

```python
# src/llamafactory/data/template.py (add after existing templates)
from .formatter import StringFormatter

register_template(
    name="newmodel",
    format_user=StringFormatter(
        slots=["<|user|>\n{{content}}<|end|>\n"]
    ),
    format_assistant=StringFormatter(
        slots=["<|assistant|>\n{{content}}<|end|>\n"]
    ),
    format_system=StringFormatter(
        slots=["<|system|>\n{{content}}<|end|>\n"]
    ),
    default_system="You are a helpful AI assistant.",
    stop_words=["<|end|>"],
    efficient_eos=True,
)
```

[View template.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/data/template.py)

### Step 3: Add Model-Specific Patches (if needed)

```python
# src/llamafactory/model/patcher.py (add to patch_model function)
def patch_model(model, model_args):
    config = model.config

    # Add NewModel-specific optimizations
    if getattr(config, "model_type", None) == "newmodel":
        if model_args.use_flash_attn:
            _patch_newmodel_attention(model)

        if model_args.rope_scaling:
            _patch_newmodel_rope(model, model_args.rope_scaling)


def _patch_newmodel_attention(model):
    """Apply FlashAttention to NewModel."""
    from .model_utils.attention import enable_flash_attention

    for layer in model.model.layers:
        layer.self_attn = enable_flash_attention(layer.self_attn)
```

[View patcher.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/model/patcher.py)

### Step 4: Handle Tokenizer Quirks

```python
# src/llamafactory/data/template.py (in get_template_and_fix_tokenizer)
def get_template_and_fix_tokenizer(tokenizer, data_args, model_args):
    template = get_template(data_args.template)

    # Fix NewModel tokenizer issues
    if data_args.template == "newmodel":
        # Add missing special tokens
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Ensure proper EOS handling
        tokenizer.eos_token = "<|end|>"

    return template
```

### Testing Your Model

```bash
# Test inference
llamafactory-cli chat \
    --model_name_or_path path/to/newmodel \
    --template newmodel

# Test training
llamafactory-cli train \
    --model_name_or_path path/to/newmodel \
    --template newmodel \
    --dataset alpaca_en \
    --output_dir ./output
```

---

## Creating Custom Datasets

### Method 1: Register in dataset_info.json

```json
// data/dataset_info.json
{
  "my_custom_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Your data file (`data/my_data.json`):
```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing uses quantum mechanical phenomena..."
  }
]
```

### Method 2: ShareGPT Format for Multi-Turn

```json
// data/dataset_info.json
{
  "my_conversations": {
    "file_name": "conversations.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

Data file:
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is Python?"},
      {"from": "gpt", "value": "Python is a programming language..."},
      {"from": "human", "value": "Show me an example"},
      {"from": "gpt", "value": "Here's a simple example:\n```python\nprint('Hello')\n```"}
    ]
  }
]
```

### Method 3: Custom Converter

For non-standard formats, create a converter:

```python
# src/llamafactory/data/converter.py
class CustomDatasetConverter(DatasetConverter):
    """Convert custom format to standard format."""

    def __call__(self, examples):
        outputs = {
            "prompt": [],
            "response": [],
            "system": [],
            "tools": [],
            "images": [],
            "videos": [],
            "audios": []
        }

        for i in range(len(examples["my_field"])):
            # Your custom conversion logic
            raw_data = examples["my_field"][i]

            # Extract prompt and response
            prompt = [{"role": "user", "content": raw_data["question"]}]
            response = [{"role": "assistant", "content": raw_data["answer"]}]

            outputs["prompt"].append(prompt)
            outputs["response"].append(response)
            outputs["system"].append(raw_data.get("context", ""))
            outputs["tools"].append("")
            outputs["images"].append([])
            outputs["videos"].append([])
            outputs["audios"].append([])

        return outputs


# Register the converter
def get_converter(dataset_attr):
    if dataset_attr.formatting == "custom":
        return CustomDatasetConverter(dataset_attr)
    # ... existing converters
```

[View converter.py](https://github.com/hiyouga/LLaMA-Factory/blob/45f0437/src/llamafactory/data/converter.py)

### Method 4: HuggingFace Hub Dataset

```json
// data/dataset_info.json
{
  "hf_dataset": {
    "hf_hub_url": "username/my-dataset",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

---

## Implementing a New Training Stage

Let's implement a hypothetical "Contrastive Learning" stage:

### Step 1: Add Stage Enum

```python
# src/llamafactory/hparams/finetuning_args.py
@dataclass
class FinetuningArguments:
    stage: Literal["pt", "sft", "rm", "dpo", "ppo", "kto", "contrastive"] = field(
        default="sft",
        metadata={"help": "Training stage"}
    )
```

### Step 2: Create Workflow

```python
# src/llamafactory/train/contrastive/workflow.py
from ..trainer_utils import create_trainer
from ...data.loader import get_dataset
from ...model.loader import load_model, load_tokenizer


def run_contrastive(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
    callbacks,
):
    """Run contrastive learning training."""
    # Load components
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Get dataset with contrastive processor
    dataset_module = get_dataset(
        template=get_template_and_fix_tokenizer(tokenizer, data_args, model_args),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="contrastive",  # Custom stage
        **tokenizer_module,
    )

    # Load model
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train,
    )

    # Create custom trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        data_collator=ContrastiveDataCollator(**tokenizer_module),
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Train
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)

    return trainer
```

### Step 3: Create Custom Trainer

```python
# src/llamafactory/train/contrastive/trainer.py
from transformers import Trainer
import torch.nn.functional as F


class ContrastiveTrainer(Trainer):
    """Trainer for contrastive learning."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get embeddings for positive and negative pairs
        anchor_outputs = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
        )
        positive_outputs = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
        )
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
        )

        # Pool to get sentence embeddings
        anchor_emb = self._pool(anchor_outputs, inputs["anchor_attention_mask"])
        positive_emb = self._pool(positive_outputs, inputs["positive_attention_mask"])
        negative_emb = self._pool(negative_outputs, inputs["negative_attention_mask"])

        # Contrastive loss
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
        loss = F.relu(neg_sim - pos_sim + self.args.margin).mean()

        return (loss, {"anchor": anchor_outputs}) if return_outputs else loss

    def _pool(self, outputs, attention_mask):
        """Mean pooling for sentence embedding."""
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1)
        return sum_embeddings / sum_mask
```

### Step 4: Add to Tuner

```python
# src/llamafactory/train/tuner.py
from .contrastive.workflow import run_contrastive

def run_exp(dict_config):
    # ... existing code ...

    if finetuning_args.stage == "contrastive":
        run_contrastive(
            model_args, data_args, training_args,
            finetuning_args, generating_args, callbacks
        )
```

---

## API Integration Patterns

### Pattern 1: OpenAI-Compatible Client

```python
import openai

# Point to LLaMA-Factory API
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Unless you configure auth
)

# Standard OpenAI usage works
response = client.chat.completions.create(
    model="llama3",  # Model name from your config
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
```

### Pattern 2: Streaming Responses

```python
# Stream responses
stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Pattern 3: Direct HTTP

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Pattern 4: Async with httpx

```python
import httpx
import asyncio

async def chat(message):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": message}],
            }
        )
        return response.json()["choices"][0]["message"]["content"]

# Use in async context
result = asyncio.run(chat("Hello!"))
```

---

## Common Extension Scenarios

### Adding a Custom Callback

```python
# my_callbacks.py
from transformers import TrainerCallback

class MetricsToPrometheusCallback(TrainerCallback):
    """Push metrics to Prometheus."""

    def __init__(self, pushgateway_url):
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        self.registry = CollectorRegistry()
        self.loss_gauge = Gauge('training_loss', 'Training loss', registry=self.registry)
        self.pushgateway_url = pushgateway_url

    def on_log(self, args, state, control, **kwargs):
        if "loss" in state.log_history[-1]:
            self.loss_gauge.set(state.log_history[-1]["loss"])
            push_to_gateway(
                self.pushgateway_url,
                job='llamafactory',
                registry=self.registry
            )


# Use it
from llamafactory.train.tuner import run_exp

config = load_config("train_config.yaml")
config["callbacks"] = [MetricsToPrometheusCallback("http://prometheus:9091")]
run_exp(config)
```

### Custom Data Processor

```python
# my_processor.py
from llamafactory.data.processor.processor_utils import DatasetProcessor

class SentimentProcessor(DatasetProcessor):
    """Process data for sentiment classification."""

    def process(self, examples):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        label_map = {"positive": 0, "negative": 1, "neutral": 2}

        for i in range(len(examples["text"])):
            text = examples["text"][i]
            label = label_map[examples["sentiment"][i]]

            # Encode
            encoded = self.tokenizer(
                text,
                max_length=self.data_args.cutoff_len,
                truncation=True,
            )

            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["attention_mask"].append(encoded["attention_mask"])
            model_inputs["labels"].append(label)

        return model_inputs
```

### Adding a New Optimization

```python
# src/llamafactory/model/model_utils/my_optimization.py
import torch.nn as nn

def apply_my_optimization(model):
    """Apply custom optimization to model."""

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "mlp" in name:
            # Replace with optimized version
            optimized = OptimizedLinear(module)
            parent = get_parent_module(model, name)
            setattr(parent, name.split(".")[-1], optimized)

    return model


class OptimizedLinear(nn.Module):
    """Memory-efficient linear layer."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        # Add your optimizations

    def forward(self, x):
        # Optimized forward pass
        pass
```

---

## Integration Testing

Always test your extensions:

```python
# tests/test_my_extension.py
import pytest
from llamafactory.data.template import get_template
from llamafactory.model.loader import load_model, load_tokenizer

def test_newmodel_template():
    """Test NewModel chat template."""
    template = get_template("newmodel")

    assert template.default_system == "You are a helpful AI assistant."
    assert "<|user|>" in template.format_user.slots[0]

def test_custom_dataset_loading():
    """Test custom dataset loads correctly."""
    from llamafactory.data.loader import load_dataset_info

    info = load_dataset_info("data")
    assert "my_custom_dataset" in info

def test_contrastive_training():
    """Test contrastive training stage."""
    # Run short training
    config = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "stage": "contrastive",
        "dataset": "my_contrastive_data",
        "max_steps": 10,
        "output_dir": "/tmp/test_contrastive",
    }

    from llamafactory.train.tuner import run_exp
    run_exp(config)

    # Verify outputs exist
    assert os.path.exists("/tmp/test_contrastive/trainer_state.json")
```

---

## Best Practices for Extensions

### Follow Existing Patterns

When extending LLaMA-Factory, follow the patterns already established:

1. **Use dataclasses for configuration**: Add new arguments to the appropriate `*_args.py` file
2. **Validate early**: Add validation in `__post_init__` or `_verify_*` functions
3. **Log appropriately**: Use `logger.info_rank0()` for distributed training
4. **Handle errors gracefully**: Provide actionable error messages

### Backwards Compatibility

When adding features, consider backwards compatibility:

```python
# Good: Optional parameter with default
def my_function(required_param, new_feature=False):
    if new_feature:
        # New behavior
        pass
    else:
        # Original behavior
        pass

# Bad: Breaking existing calls
def my_function(required_param, new_required_param):  # Breaks existing code
    pass
```

### Documentation Requirements

For any extension to be merged upstream, include:
- Module and function docstrings (Google style)
- Type hints for all parameters and returns
- Example in docstring or tests
- Update relevant README sections

---

## Troubleshooting Common Issues

### Model Not Loading

```python
# Error: "Model architecture not found"
# Solution: Add to SUPPORTED_MODELS in constants.py
```

### Template Not Applied

```python
# Check: Is template registered?
from llamafactory.data.template import TEMPLATES
print("mytemplate" in TEMPLATES)  # Should be True

# Check: Is template specified in config?
# --template mytemplate
```

### Custom Dataset Not Found

```python
# Check: Is dataset in registry?
from llamafactory.data.loader import load_dataset_info
info = load_dataset_info("data")
print("my_dataset" in info)

# Check: Does file exist at specified path?
```

### Training Stage Not Recognized

```python
# Check: Is stage added to FinetuningArguments?
# Check: Is workflow imported in tuner.py?
# Check: Is routing added to run_exp()?
```

---

## Key Takeaways

1. **Models**: Add to constants, create template, patch if needed
2. **Datasets**: Register in dataset_info.json or create custom converter
3. **Training Stages**: Create workflow, trainer, and register in tuner
4. **API Integration**: Use OpenAI-compatible client for easy migration
5. **Custom Callbacks**: Extend TrainerCallback for monitoring/logging
6. **Testing**: Always create tests for your extensions

---

## What's Next

In [Blog 6: Performance Analysis and Optimization](./06-performance-analysis.md), we'll analyze LLaMA-Factory's performance characteristics and explore optimization opportunities.

---

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
