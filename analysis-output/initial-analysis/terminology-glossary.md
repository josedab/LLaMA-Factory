# Terminology Glossary

**Analysis Commit SHA:** `45f0437`

---

## Training Stages

### PT (Pre-Training)
Continued pre-training on domain-specific data. Used to adapt base models to new domains without instruction tuning.

**Related files**: `src/llamafactory/train/pt/workflow.py`

### SFT (Supervised Fine-Tuning)
Instruction fine-tuning where models learn to follow instructions from input-output pairs.

**Related files**: `src/llamafactory/train/sft/workflow.py`

### RM (Reward Modeling)
Training a reward model to score responses based on human preferences. Used as part of RLHF pipeline.

**Related files**: `src/llamafactory/train/rm/workflow.py`

### DPO (Direct Preference Optimization)
A simpler alternative to PPO that directly optimizes the policy from preference data without an explicit reward model.

**Related files**: `src/llamafactory/train/dpo/workflow.py`

### PPO (Proximal Policy Optimization)
Reinforcement learning algorithm for fine-tuning language models using human feedback (RLHF).

**Related files**: `src/llamafactory/train/ppo/workflow.py`

### KTO (Kahneman-Tversky Optimization)
Preference optimization based on prospect theory, using binary feedback (good/bad) rather than pairwise comparisons.

**Related files**: `src/llamafactory/train/kto/workflow.py`

---

## Adapter Methods

### LoRA (Low-Rank Adaptation)
Fine-tuning method that adds trainable low-rank matrices to frozen model weights. Significantly reduces memory and training time.

**Key parameters**: `lora_rank`, `lora_alpha`, `lora_target`

### QLoRA (Quantized LoRA)
Combines 4-bit quantization with LoRA for memory-efficient fine-tuning of large models.

**Key parameters**: `quantization_bit: 4`, `quantization_method: "bitsandbytes"`

### OFT (Orthogonal Fine-Tuning)
Adapter method using orthogonal transformations to preserve pre-trained features.

### DoRA (Weight-Decomposed Low-Rank Adaptation)
Extension of LoRA that separates magnitude and direction in weight updates.

### PiSSA (Principal Singular values and Singular vectors Adaptation)
Uses SVD-based initialization for LoRA adapters.

### LLaMA Pro
Method for adding new blocks to expand model capacity during fine-tuning.

---

## Optimization Techniques

### FlashAttention-2
Memory-efficient attention algorithm that reduces memory usage from O(N²) to O(N).

### GaLore (Gradient Low-Rank Projection)
Optimizer that projects gradients into low-rank space to reduce memory.

**Related files**: `src/llamafactory/train/trainer_utils.py`

### APOLLO
Advanced optimizer with adaptive learning rate based on gradient statistics.

### BAdam (Block-wise Adam)
Adam optimizer variant that updates parameters in blocks for memory efficiency.

### Adam-mini
Compact Adam variant with reduced memory footprint.

### Muon Optimizer
Custom optimizer in third-party directory.

**Related files**: `src/llamafactory/third_party/muon/`

---

## Quantization Methods

### BNB (BitsAndBytes)
Library for 8-bit and 4-bit quantization of neural network weights.

### GPTQ
Post-training quantization using approximate second-order information.

### AWQ (Activation-aware Weight Quantization)
Quantization method that protects salient weights based on activation patterns.

### AQLM (Additive Quantization of Language Models)
Codebook-based quantization for extreme compression.

### EETQ
Efficient Equal-Task Quantization method.

### HQQ (Half-Quadratic Quantization)
Fast quantization method without calibration data.

### Quanto
Quantization toolkit for PyTorch models.

### MXFP4
Mixed-precision FP4 quantization format.

---

## Inference Engines

### HuggingFace Engine
Default inference using transformers library's generation methods.

**Related files**: `src/llamafactory/chat/hf_engine.py`

### vLLM
High-throughput inference engine using PagedAttention for efficient KV-cache management.

**Related files**: `src/llamafactory/chat/vllm_engine.py`

### SGLang
Inference engine with speculative decoding and RadixAttention for fast serving.

**Related files**: `src/llamafactory/chat/sglang_engine.py`

### KTransformers
CPU-optimized inference engine for running LLMs without GPU.

**Related files**: `src/llamafactory/chat/kt_engine.py`

---

## Data Formats

### Alpaca Format
Simple instruction format with `instruction`, `input`, and `output` fields.

```json
{
  "instruction": "Summarize the following text",
  "input": "Long text here...",
  "output": "Summary here..."
}
```

### ShareGPT Format
Multi-turn conversation format with role-based messages.

```json
{
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"}
  ]
}
```

### OpenAI Format
Standard chat format with system/user/assistant roles.

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

---

## Configuration Terms

### finetuning_type
The adapter method to use: `"lora"`, `"oft"`, `"full"`, `"freeze"`

### stage
Training stage: `"pt"`, `"sft"`, `"rm"`, `"dpo"`, `"ppo"`, `"kto"`

### template
Chat template for formatting conversations. Examples: `"llama3"`, `"qwen"`, `"mistral"`

### dataset_info.json
Registry file mapping dataset names to their configurations and file locations.

### neat_packing
Efficient sequence packing that avoids cross-contamination between samples.

### train_on_prompt
Whether to include prompt in loss calculation (default: False).

---

## Hardware Terms

### CUDA
NVIDIA's parallel computing platform for GPU acceleration.

### ROCm
AMD's GPU computing platform, alternative to CUDA.

### NPU (Neural Processing Unit)
Huawei Ascend AI processors for neural network acceleration.

### RDZV (Rendezvous)
PyTorch's mechanism for coordinating distributed training processes.

---

## Distributed Training Terms

### LOCAL_RANK
Rank of the current process on the local node (0 to num_gpus-1).

### WORLD_SIZE
Total number of processes across all nodes.

### torchrun
PyTorch's distributed training launcher (replacement for torch.distributed.launch).

### DeepSpeed
Microsoft's distributed training library for memory optimization and parallelism.

### MCA (Megatron Core Adapter)
Integration with NVIDIA's Megatron framework for large-scale training.

---

## API Terms

### OpenAI-compatible API
REST API following OpenAI's chat completion specification for drop-in compatibility.

### SSE (Server-Sent Events)
Protocol for streaming responses from server to client.

### SSRF (Server-Side Request Forgery)
Security vulnerability where server can be tricked into making requests to internal resources.

### LFI (Local File Inclusion)
Security vulnerability where server can be tricked into accessing local files.

---

## Model Terms

### ValueHead
Additional head for predicting values in RM and PPO training.

**Related files**: `src/llamafactory/model/model_utils/valuehead.py`

### RMSNorm
Root Mean Square Layer Normalization, used in LLaMA models.

### RoPE (Rotary Position Embedding)
Position encoding method used in modern LLMs.

### Shift-Short Attention (S²-Attn)
Attention pattern for efficient long-context training.

---

## Web UI Terms

### LLaMA Board
The Gradio-based web interface for LLaMA-Factory.

### Manager
Resource manager handling model loading and unloading in WebUI.

**Related files**: `src/llamafactory/webui/manager.py`

---

## File Naming Conventions

### workflow.py
Training workflow implementation for each stage.

### *_args.py
Dataclass definitions for argument groups.

### *_engine.py
Inference engine implementations.

### *_utils.py
Utility functions for a module.

### template.py
Chat/prompt template definitions.

### constants.py
Static definitions (supported models, etc.).

---

## Environment Variables

### USE_V1
Enable experimental V1 API (`"1"` to enable).

### FORCE_TORCHRUN
Force distributed training launcher (`"1"` to enable).

### USE_MODELSCOPE_HUB
Use ModelScope instead of HuggingFace Hub (`"1"` to enable).

### LLAMAFACTORY_VERBOSITY
Logging level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`

### LLAMABOARD_ENABLED
WebUI mode indicator (set automatically).

### RECORD_VRAM
Enable VRAM tracking during training (`"1"` to enable).

### API_VERBOSE
Enable verbose API request logging (`"0"` or `"1"`).

### HF_TOKEN
HuggingFace access token for gated models.

---

## Abbreviations

| Abbrev | Full Name |
|--------|-----------|
| LLM | Large Language Model |
| RLHF | Reinforcement Learning from Human Feedback |
| HF | HuggingFace |
| PT | Pre-Training |
| SFT | Supervised Fine-Tuning |
| RM | Reward Model/Modeling |
| DPO | Direct Preference Optimization |
| PPO | Proximal Policy Optimization |
| KTO | Kahneman-Tversky Optimization |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| FA | FlashAttention |
| KV | Key-Value (cache) |
| BNB | BitsAndBytes |
| LOC | Lines of Code |
