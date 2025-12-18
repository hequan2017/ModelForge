# DeepSpeed ZeRO（零冗余优化器）

## 简单理解

**DeepSpeed ZeRO** 是一种分布式训练技术，把模型的"负担"分摊到多张 GPU 上，让每张卡只存一部分，从而训练更大的模型。

---

## 类比解释

```
没有 ZeRO（数据并行）：
┌─────────────────────────────────────────────────┐
│ 每个人都背着完整的行李（模型+梯度+优化器）        │
│                                                 │
│ GPU 0: 🎒🎒🎒  GPU 1: 🎒🎒🎒  GPU 2: 🎒🎒🎒     │
│ (完整)        (完整)        (完整)              │
│                                                 │
│ 问题：每个人都累死了（显存爆炸）                  │
└─────────────────────────────────────────────────┘

有 ZeRO（零冗余）：
┌─────────────────────────────────────────────────┐
│ 大家分工，每人只背一部分                         │
│                                                 │
│ GPU 0: 🎒    GPU 1: 🎒    GPU 2: 🎒             │
│ (1/3)       (1/3)       (1/3)                  │
│                                                 │
│ 需要时互相借用，用完还回去                       │
└─────────────────────────────────────────────────┘
```

---

## 显存占用分析

训练一个模型需要存储：

```
以 8B 参数模型为例（bfloat16）：

1. 模型参数（Parameters）
   8B × 2 bytes = 16 GB

2. 梯度（Gradients）
   8B × 2 bytes = 16 GB

3. 优化器状态（Optimizer States）
   Adam 需要存储：
   - 一阶动量（m）：8B × 4 bytes = 32 GB
   - 二阶动量（v）：8B × 4 bytes = 32 GB
   总计：64 GB

总显存需求：16 + 16 + 64 = 96 GB（单卡根本装不下！）
```

---

## ZeRO 三个阶段

### ZeRO-1：分片优化器状态

```
原来（每卡都存完整优化器状态）：
GPU 0: [参数][梯度][优化器状态 64GB] 
GPU 1: [参数][梯度][优化器状态 64GB]  ← 重复存储
GPU 2: [参数][梯度][优化器状态 64GB]

ZeRO-1（优化器状态分片）：
GPU 0: [参数][梯度][优化器 1/3]  ← 只存 21GB
GPU 1: [参数][梯度][优化器 1/3]
GPU 2: [参数][梯度][优化器 1/3]

节省：64GB → 21GB/卡（节省约 4 倍）
```

### ZeRO-2：分片优化器状态 + 梯度

```
ZeRO-2（优化器状态 + 梯度分片）：
GPU 0: [参数][梯度 1/3][优化器 1/3]
GPU 1: [参数][梯度 1/3][优化器 1/3]
GPU 2: [参数][梯度 1/3][优化器 1/3]

节省：(64+16)GB → 27GB/卡（节省约 8 倍）
```

### ZeRO-3：分片一切

```
ZeRO-3（参数 + 梯度 + 优化器状态全部分片）：
GPU 0: [参数 1/3][梯度 1/3][优化器 1/3]
GPU 1: [参数 1/3][梯度 1/3][优化器 1/3]
GPU 2: [参数 1/3][梯度 1/3][优化器 1/3]

节省：(16+16+64)GB → 32GB/卡（节省约 16 倍）
```

---

## 图解对比

```
                    ZeRO-1      ZeRO-2      ZeRO-3
                    ───────     ───────     ───────
模型参数            完整        完整        分片 ✓
梯度                完整        分片 ✓      分片 ✓
优化器状态          分片 ✓      分片 ✓      分片 ✓

显存节省            ~4x         ~8x         ~16x
通信开销            低          中          高
训练速度            快          较快        较慢
```

---

## 显存计算示例

### 8B 模型，5 张 GPU

| 阶段 | 参数 | 梯度 | 优化器 | 总计/卡 |
|------|------|------|--------|--------|
| 无 ZeRO | 16GB | 16GB | 64GB | 96GB ❌ |
| ZeRO-1 | 16GB | 16GB | 13GB | 45GB ❌ |
| ZeRO-2 | 16GB | 3GB | 13GB | 32GB ✓ |
| ZeRO-3 | 3GB | 3GB | 13GB | 19GB ✓ |

（还需要加上激活值的显存，实际会更高）

---

## ZeRO-Offload（CPU 卸载）

当 GPU 显存还是不够时，可以把部分数据卸载到 CPU 内存：

```
ZeRO-3 + Offload：
┌─────────────────────────────────────────────────┐
│ GPU 显存：只放计算必需的数据                      │
│ CPU 内存：存放优化器状态、部分参数                │
│                                                 │
│ GPU ←──────→ CPU                               │
│     │ 需要时加载 │                              │
│     │ 用完卸载   │                              │
└─────────────────────────────────────────────────┘

优点：可以训练更大的模型
缺点：CPU-GPU 数据传输会拖慢训练速度
```

---

## 使用方法

### Swift 中使用

```bash
# ZeRO-1
--deepspeed zero1

# ZeRO-2（推荐，平衡显存和速度）
--deepspeed zero2

# ZeRO-3（显存紧张时使用）
--deepspeed zero3

# ZeRO-3 + CPU Offload（显存非常紧张）
--deepspeed zero3_offload
```

### 自定义 DeepSpeed 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true
  },
  "bf16": {
    "enabled": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

---

## 选择建议

```
模型大小          推荐配置
─────────────────────────────────────
< 7B             不需要 DeepSpeed 或 ZeRO-1
7B - 13B         ZeRO-2
13B - 30B        ZeRO-2 或 ZeRO-3
30B - 70B        ZeRO-3
> 70B            ZeRO-3 + Offload
```

### 根据显存选择

```
单卡显存    模型大小    推荐配置
──────────────────────────────────────
24GB       7B         ZeRO-2 + LoRA
40GB       8B         ZeRO-2 全参数
40GB       30B        ZeRO-3 + LoRA
80GB       30B        ZeRO-2 全参数
80GB       70B        ZeRO-3 + LoRA
```

---

## 通信开销分析

```
ZeRO-1：
前向传播：无额外通信
反向传播：Reduce-Scatter 梯度
参数更新：All-Gather 参数

ZeRO-2：
前向传播：无额外通信
反向传播：Reduce-Scatter 梯度（分片存储）
参数更新：All-Gather 参数

ZeRO-3：
前向传播：All-Gather 参数（需要时获取）
反向传播：Reduce-Scatter 梯度
参数更新：All-Gather 参数

通信量：ZeRO-1 < ZeRO-2 < ZeRO-3
```

---

## 实际案例

### 案例 1：8B 模型全参数微调

```bash
# 5 张 44GB GPU
# ZeRO-2 显存不够，需要 ZeRO-3

torchrun --nproc_per_node=5 swift sft \
  --model Qwen/Qwen3-8B \
  --train_type full \
  --deepspeed zero3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16
```

### 案例 2：30B 模型 LoRA 微调

```bash
# 5 张 44GB GPU
# LoRA + ZeRO-2 够用

torchrun --nproc_per_node=5 swift sft \
  --model Qwen/Qwen3-30B-A3B \
  --train_type lora \
  --deepspeed zero2 \
  --per_device_train_batch_size 2
```

---

## 常见问题

### Q: ZeRO-3 为什么比 ZeRO-2 慢？

```
ZeRO-3 在前向传播时也需要通信：
1. 计算某一层前，先 All-Gather 该层参数
2. 计算完成后，释放参数显存
3. 每一层都要重复这个过程

通信开销增加，但换来了更低的显存占用
```

### Q: 什么时候用 Offload？

```
当 ZeRO-3 还是 OOM 时：
1. 模型太大（>70B）
2. GPU 显存太小
3. 序列长度太长

注意：Offload 会显著降低训练速度（2-3 倍）
```

### Q: 如何判断用哪个 ZeRO 阶段？

```
1. 先试 ZeRO-2（速度最快）
2. OOM 了换 ZeRO-3
3. 还 OOM 就用 ZeRO-3 + Offload
4. 或者改用 LoRA 微调
```

---

## 与训练参数的关系

| 参数 | 说明 |
|------|------|
| `--deepspeed zero1` | ZeRO Stage 1 |
| `--deepspeed zero2` | ZeRO Stage 2（推荐） |
| `--deepspeed zero3` | ZeRO Stage 3 |
| `--deepspeed zero3_offload` | ZeRO-3 + CPU Offload |

---

## 一句话总结

> **DeepSpeed ZeRO** 是"分担负重"的技术——把模型的参数、梯度、优化器状态分摊到多张 GPU，让小显存也能训练大模型。

