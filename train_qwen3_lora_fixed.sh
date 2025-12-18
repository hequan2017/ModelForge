#!/bin/bash

# ============================================================
# 方案一：使用 ZeRO-3（推荐）
# ZeRO-3 会在模型加载阶段就分片，解决 OOM 问题
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  --no_python swift sft \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
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
  --output_dir output_qwen3_30b_full \
  --system 'You are a helpful assistant.' \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name swift-robot \
  --deepspeed zero3

