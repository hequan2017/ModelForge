#!/bin/bash

# ============================================================
# Qwen3-8B 全参数微调脚本
# 5x 44GB GPU - 使用 ZeRO-3 减少显存占用
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  --no_python swift sft \
  --model Qwen/Qwen3-8B \
  --train_type full \
  --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k#1000' \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --gradient_checkpointing true \
  --max_length 2048 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir output_qwen3_8b_full \
  --system 'You are a helpful assistant.' \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name swift-robot \
  --deepspeed zero3
