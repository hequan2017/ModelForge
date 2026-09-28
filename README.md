[简体中文](README.md) | [English](README.en.md)

# ModelForge

基于 ms-swift 的大模型微调实战仓库：Qwen3 LoRA 训练 / 推理脚本，配套逐参数讲解、基础知识文档与完整八步流程指南（含 Qwen3-30B-A3B 全参数微调示例）。

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![ms-swift](https://img.shields.io/badge/ms--swift-魔搭训练框架-7C3AED?style=flat)](https://github.com/modelscope/ms-swift)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO--2-0078D4?style=flat)](https://github.com/microsoft/DeepSpeed)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

## ✨ 仓库内容

- **训练 / 推理脚本**（5 x 44GB GPU，Qwen3-8B LoRA，多机多卡用 `torchrun` 拉起）：
  - `train_qwen3_lora.sh` —— 标准版：rank 64 / alpha 128，取 10000 条数据训练 3 个 epoch
  - `train_qwen3_lora_fixed.sh` —— 快速版：rank 128 / alpha 256，取 1000 条数据快速跑通验证
  - `infer_qwen3_lora.sh` —— 推理：自动定位最新 checkpoint，流式输出对话
- **`train_qwen3_lora_参数说明.md`** —— 对脚本中每个参数（CUDA_VISIBLE_DEVICES、lora_rank、deepspeed 等）的详细解释与调优建议
- **`大模型基础知识/`** —— 12 篇基础概念文档（梯度、损失函数、学习率、Batch Size、LoRA、DeepSpeed ZeRO、Transformer、混合精度等），按入门/进阶学习路线组织
- **八步流程指南**（见下文）—— 环境准备 → 数据准备 → 配置检查 → 开始训练 → 监控训练 → 模型合并 → 模型测试 → 模型部署

## 🛠 技术栈

- 训练框架：[ms-swift](https://github.com/modelscope/ms-swift)（`swift sft`）+ torchrun（5 卡）
- 分布式加速：DeepSpeed ZeRO-2，bfloat16 混合精度，gradient checkpointing
- 模型：Qwen/Qwen3-8B（指南部分示例为 Qwen3-30B-A3B-Thinking-2507 全参数微调）
- 数据集：`liucong/Chinese-DeepSeek-R1-Distill-data-110k`（支持 JSONL / Alpaca 格式本地数据）

## 🚀 快速开始

```bash
# 1. 安装环境（CUDA >= 11.8）
conda create -n swift python=3.10 -y && conda activate swift
pip install 'ms-swift[llm]' -U
pip install deepspeed -U

# 2. 启动训练（推荐后台运行，断开 SSH 不中断）
chmod +x train_qwen3_lora.sh
nohup bash train_qwen3_lora.sh > train.log 2>&1 &
tail -f train.log            # 观察 loss / epoch / global_step

# 3. 推理验证（自动加载 output_qwen3_8b_lora 下最新 checkpoint）
bash infer_qwen3_lora.sh

# 4. 部署为 OpenAI 兼容 API
swift deploy --model output_qwen3_30b_full/checkpoint-best --port 8000
```

显存不足时可减小 `per_device_train_batch_size`、`max_length`，或改用 `zero3_offload`；模型下载慢可设置 `HF_ENDPOINT=https://hf-mirror.com`。更多细节见《参数说明》文档与流程指南。

## 📁 目录结构

```
ModelForge/
├── train_qwen3_lora.sh          # LoRA 训练脚本（标准版）
├── train_qwen3_lora_fixed.sh    # LoRA 训练脚本（快速版）
├── infer_qwen3_lora.sh          # LoRA 推理脚本
├── train_qwen3_lora_参数说明.md  # 训练参数逐项讲解
└── 大模型基础知识/               # 12 篇训练基础知识文档
```

## 📄 License

本项目基于 [Apache-2.0](LICENSE) 协议开源。

## 🔗 相关项目

- 作者主页：[hequan2017](https://github.com/hequan2017)
