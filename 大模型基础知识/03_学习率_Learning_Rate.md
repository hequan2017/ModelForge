# 学习率（Learning Rate）

## 简单理解

**学习率**就是"步长"，决定了每次参数更新时走多大一步。太大会跳过最优点，太小会走得很慢。

---

## 类比解释

想象你在山上找最低点：

```
学习率太大：
    ●──────────→──────────→ 跳来跳去，永远找不到
   ╱ ╲        ╱ ╲
  ╱   ╲      ╱   ╲
 ╱     ╲    ╱     ╲
        ★

学习率太小：
    ●→→→→→→→→→→→→→→ 走得太慢，训练不完
   ╱ ╲
  ╱   ╲
 ╱     ★

学习率合适：
    ●──→──→──→★  几步就到达最低点
   ╱ ╲
  ╱   ╲
 ╱     ╲
```

---

## 参数更新公式

```
新参数 = 旧参数 - 学习率 × 梯度

例如：
- 旧参数 w = 2.0
- 梯度 = -6.0（表示应该增大）
- 学习率 = 0.1

新参数 = 2.0 - 0.1 × (-6.0) = 2.0 + 0.6 = 2.6
```

---

## 学习率的影响

| 学习率 | 效果 | 表现 |
|--------|------|------|
| **太大** (1e-3) | Loss 震荡或爆炸 | 训练不稳定，Loss 上下跳动 |
| **太小** (1e-7) | 收敛极慢 | 训练很久 Loss 几乎不变 |
| **合适** (1e-5) | 平稳收敛 | Loss 稳定下降 |

---

## 不同任务的推荐学习率

| 任务类型 | 推荐学习率 | 说明 |
|---------|-----------|------|
| 预训练 | 1e-4 ~ 3e-4 | 从零学习，可以大一些 |
| 全参数微调 | 1e-5 ~ 5e-5 | 保护预训练知识，要小 |
| LoRA 微调 | 1e-4 ~ 5e-4 | 只训练适配器，可以大一些 |
| 推理/评估 | 0 | 不更新参数 |

---

## 学习率调度（Learning Rate Schedule）

训练过程中，学习率通常不是固定的，而是动态变化的：

### 1. 预热（Warmup）

```
学习率
  │        ╭────────
  │       ╱
  │      ╱
  │     ╱
  │    ╱
  └───┴──────────────→ Steps
      预热期
```

- 训练初期从 0 逐渐增加到设定值
- 避免初期大学习率导致的不稳定
- `--warmup_ratio 0.1` 表示前 10% 的步数用于预热

### 2. 余弦退火（Cosine Annealing）

```
学习率
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲___
  └────────────────→ Steps
```

- 学习率按余弦曲线逐渐减小
- 训练后期用小学习率精细调整

### 3. 完整曲线

```
学习率
  │      ╭──────────╮
  │     ╱            ╲
  │    ╱              ╲
  │   ╱                ╲
  │  ╱                  ╲
  └─┴────────────────────→ Steps
    预热    正常训练    衰减
```

---

## 代码示例

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 创建模型和优化器
model = torch.nn.Linear(10, 2)
optimizer = AdamW(model.parameters(), lr=1e-4)  # 设置学习率

# 创建学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 训练循环
for step in range(100):
    # ... 训练代码 ...
    optimizer.step()
    scheduler.step()  # 更新学习率
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Step {step}: lr = {current_lr:.6f}")
```

---

## 学习率调试技巧

### 1. 学习率探测（LR Finder）

```
Loss
  │
  │        ╲
  │         ╲___╱
  │              ╲
  │               ╲
  └──────────────────→ 学习率 (log scale)
     1e-7    1e-5   1e-3
            ↑
        最佳学习率（Loss 下降最快的点）
```

### 2. 经验法则

```python
# 基础学习率
base_lr = 1e-4

# 根据 batch size 调整（线性缩放）
# batch size 翻倍，学习率也可以翻倍
adjusted_lr = base_lr * (batch_size / 32)

# 根据 GPU 数量调整
adjusted_lr = base_lr * num_gpus
```

---

## 常见问题

### Loss 爆炸（变成 NaN）

```
原因：学习率太大
解决：减小学习率，如 1e-5 → 1e-6
```

### Loss 下降太慢

```
原因：学习率太小
解决：增大学习率，如 1e-6 → 1e-5
```

### Loss 震荡

```
原因：学习率太大或 batch size 太小
解决：
1. 减小学习率
2. 增大 batch size 或 gradient_accumulation_steps
3. 增大 warmup_ratio
```

---

## 与训练参数的关系

| 参数 | 与学习率的关系 |
|------|---------------|
| `--learning_rate` | 直接设置学习率 |
| `--warmup_ratio` | 预热阶段占比 |
| `--per_device_train_batch_size` | batch 大可以用更大学习率 |
| `--gradient_accumulation_steps` | 等效增大 batch，可以用更大学习率 |

---

## 一句话总结

> **学习率**决定了模型学习的"步伐"——太大会乱跳，太小会龟速，合适才能高效到达最优点。

