#!/bin/bash

# ============================================================
# Qwen3-8B LoRA 推理脚本
# 使用训练好的 LoRA 适配器进行对话
# ============================================================

# 获取最新的检查点目录
OUTPUT_DIR="output_qwen3_8b_lora"
LATEST_VERSION=$(ls -t /data/${OUTPUT_DIR}/ | head -1)
LATEST_CHECKPOINT=$(ls -t /data/${OUTPUT_DIR}/${LATEST_VERSION}/ | grep checkpoint | head -1)

echo "使用模型: /data/${OUTPUT_DIR}/${LATEST_VERSION}/${LATEST_CHECKPOINT}"

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen3-8B \
    --adapters /data/${OUTPUT_DIR}/${LATEST_VERSION}/${LATEST_CHECKPOINT} \
    --stream true \
    --temperature 0.7 \
    --max_new_tokens 2048

