[简体中文](README.md) | [English](README.en.md)

# ModelForge

A hands-on LLM fine-tuning repository based on ms-swift: Qwen3 LoRA training / inference scripts, accompanied by parameter-by-parameter explanations, fundamentals documents, and a complete eight-step workflow guide (including a Qwen3-30B-A3B full fine-tuning example).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![ms-swift](https://img.shields.io/badge/ms--swift-Training_Framework-7C3AED?style=flat)](https://github.com/modelscope/ms-swift)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO--2-0078D4?style=flat)](https://github.com/microsoft/DeepSpeed)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

## ✨ What's Inside

- **Training / inference scripts** (5 x 44GB GPUs, Qwen3-8B LoRA, launched with `torchrun`):
  - `train_qwen3_lora.sh` — standard version: rank 64 / alpha 128, 10,000 samples for 3 epochs
  - `train_qwen3_lora_fixed.sh` — fast version: rank 128 / alpha 256, 1,000 samples for a quick end-to-end run
  - `infer_qwen3_lora.sh` — inference: automatically picks the latest checkpoint and chats with streaming output
- **`train_qwen3_lora_参数说明.md`** — detailed explanations and tuning tips for every parameter in the scripts (CUDA_VISIBLE_DEVICES, lora_rank, deepspeed, etc.)
- **`大模型基础知识/`** — 12 fundamentals documents (gradient, loss, learning rate, batch size, LoRA, DeepSpeed ZeRO, Transformer, mixed precision, etc.), organized into beginner and advanced learning paths
- **Eight-step workflow guide** (see the original guide) — environment setup → data preparation → config check → training → monitoring → model merging → testing → deployment

## 🛠 Tech Stack

- Training framework: [ms-swift](https://github.com/modelscope/ms-swift) (`swift sft`) + torchrun (5 GPUs)
- Distributed acceleration: DeepSpeed ZeRO-2, bfloat16 mixed precision, gradient checkpointing
- Models: Qwen/Qwen3-8B (the guide's example covers full fine-tuning of Qwen3-30B-A3B-Thinking-2507)
- Dataset: `liucong/Chinese-DeepSeek-R1-Distill-data-110k` (local JSONL / Alpaca-format data supported)

## 🚀 Quick Start

```bash
# 1. Set up the environment (CUDA >= 11.8)
conda create -n swift python=3.10 -y && conda activate swift
pip install 'ms-swift[llm]' -U
pip install deepspeed -U

# 2. Start training (running in the background is recommended so it survives SSH disconnects)
chmod +x train_qwen3_lora.sh
nohup bash train_qwen3_lora.sh > train.log 2>&1 &
tail -f train.log            # Watch loss / epoch / global_step

# 3. Verify with inference (auto-loads the latest checkpoint under output_qwen3_8b_lora)
bash infer_qwen3_lora.sh

# 4. Deploy as an OpenAI-compatible API
swift deploy --model output_qwen3_30b_full/checkpoint-best --port 8000
```

If GPU memory runs short, reduce `per_device_train_batch_size` and `max_length`, or switch to `zero3_offload`; for slow model downloads set `HF_ENDPOINT=https://hf-mirror.com`. See the parameter-explanation document and the workflow guide for details.

## 📁 Directory Structure

```
ModelForge/
├── train_qwen3_lora.sh          # LoRA training script (standard)
├── train_qwen3_lora_fixed.sh    # LoRA training script (fast)
├── infer_qwen3_lora.sh          # LoRA inference script
├── train_qwen3_lora_参数说明.md  # Parameter-by-parameter explanations
└── 大模型基础知识/               # 12 training fundamentals documents
```

## 📄 License

This project is licensed under [Apache-2.0](LICENSE).

## 🔗 Related Projects

- Author profile: [hequan2017](https://github.com/hequan2017)
