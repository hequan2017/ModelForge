#!/bin/bash

# ============================================================
# Qwen3-8B LoRA 微调脚本
# 5x 44GB GPU - 8B 模型可以用更大 batch 加速训练
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  --no_python swift sft \
  --model Qwen/Qwen3-8B \
  --train_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k#1000' \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --gradient_checkpointing true \
  --max_length 4096 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir output_qwen3_8b_lora \
  --system 'You are a helpful assistant.' \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name swift-robot \
  --deepspeed zero2
