#!/bin/bash

# ============================================================
# 方案：全参数微调 + CPU Offload
# 如果一定要全参数微调，必须启用 CPU Offload
# 会将部分优化器状态和参数卸载到 CPU 内存
# 注意：训练速度会明显变慢
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --max_length 1024 \
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
  --deepspeed zero3_offload

