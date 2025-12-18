# 优化器（Optimizer）

## 简单理解

**优化器**是"教练"，根据梯度（错误方向）来指导参数如何更新。不同的优化器有不同的"教学方法"。

---

## 类比解释

```
你在山上找最低点（最优参数）：

SGD（随机梯度下降）：
┌─────────────────────────────────────┐
│ 😊 "梯度指哪，我就往哪走"             │
│ 简单直接，但容易卡在小坑里            │
└─────────────────────────────────────┘

Adam（自适应矩估计）：
┌─────────────────────────────────────┐
│ 🧠 "我会记住之前的方向，聪明地调整"    │
│ 考虑历史信息，走得更稳更快            │
└─────────────────────────────────────┘
```

---

## 参数更新基本公式

```
新参数 = 旧参数 - 学习率 × 梯度

θ_new = θ_old - lr × ∇L

其中：
- θ：模型参数
- lr：学习率
- ∇L：损失函数的梯度
```

---

## 常见优化器

### 1. SGD（随机梯度下降）

```
最简单的优化器：

θ = θ - lr × g

特点：
✅ 简单，计算快
❌ 容易陷入局部最优
❌ 对学习率敏感
```

### 2. SGD + Momentum（动量）

```
加入"惯性"，像滚球一样：

v = β × v + g           # 速度 = 历史速度 + 当前梯度
θ = θ - lr × v          # 参数更新

β 通常设为 0.9

特点：
✅ 加速收敛
✅ 减少震荡
❌ 多了一个超参数
```

```
图解：

没有动量：
    ●─→─→─→─→─→─→  走走停停，方向变来变去
   ╱ ╲
  ╱   ★

有动量：
    ●────────────→  一路冲向目标
   ╱ ╲
  ╱   ★
```

### 3. Adam（Adaptive Moment Estimation）⭐

```
结合了动量和自适应学习率，是目前最常用的优化器：

m = β1 × m + (1-β1) × g        # 一阶矩（动量）
v = β2 × v + (1-β2) × g²       # 二阶矩（梯度平方）
m̂ = m / (1-β1^t)               # 偏差修正
v̂ = v / (1-β2^t)
θ = θ - lr × m̂ / (√v̂ + ε)     # 参数更新

默认参数：
- β1 = 0.9
- β2 = 0.999
- ε = 1e-8
```

**Adam 的优势**：

```
自适应学习率：
┌─────────────────────────────────────┐
│ 梯度大的参数 → 学习率自动变小         │
│ 梯度小的参数 → 学习率自动变大         │
│                                     │
│ 每个参数都有"个性化"的学习率          │
└─────────────────────────────────────┘
```

### 4. AdamW（Adam with Weight Decay）⭐

```
Adam + 权重衰减（正则化）：

θ = θ - lr × (m̂ / (√v̂ + ε) + λ × θ)
                              ↑
                         权重衰减项

λ 通常设为 0.01

特点：
✅ 防止过拟合
✅ 大模型训练的首选
```

---

## 优化器对比

| 优化器 | 速度 | 稳定性 | 显存 | 适用场景 |
|--------|------|--------|------|---------|
| SGD | 快 | 一般 | 低 | 简单任务 |
| SGD+Momentum | 快 | 较好 | 低 | 计算机视觉 |
| Adam | 较快 | 好 | 高 | 大多数任务 |
| AdamW | 较快 | 好 | 高 | 大模型微调 |

---

## 优化器显存占用

```
Adam/AdamW 需要存储额外状态：

模型参数：8B × 2 bytes = 16 GB
一阶矩 m：8B × 4 bytes = 32 GB
二阶矩 v：8B × 4 bytes = 32 GB
─────────────────────────────
优化器状态总计：64 GB

这就是为什么训练比推理需要更多显存！
```

---

## 代码示例

### PyTorch 中使用优化器

```python
import torch
from torch.optim import Adam, AdamW, SGD

# 创建模型
model = torch.nn.Linear(100, 10)

# SGD 优化器
optimizer_sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam 优化器
optimizer_adam = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# AdamW 优化器（推荐用于大模型）
optimizer_adamw = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 训练循环
for epoch in range(100):
    # 前向传播
    output = model(input_data)
    loss = criterion(output, target)
    
    # 反向传播
    optimizer_adamw.zero_grad()  # 清空梯度
    loss.backward()              # 计算梯度
    optimizer_adamw.step()       # 更新参数
```

### Transformers 中设置优化器

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=1e-5,          # 学习率
    weight_decay=0.01,           # 权重衰减
    adam_beta1=0.9,              # Adam β1
    adam_beta2=0.999,            # Adam β2
    adam_epsilon=1e-8,           # Adam ε
    optim="adamw_torch",         # 优化器类型
)
```

---

## 学习率调度

优化器通常配合学习率调度器使用：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

# 线性预热
scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=100)

# 训练循环中
for step in range(total_steps):
    optimizer.step()
    scheduler.step()  # 更新学习率
```

---

## 梯度裁剪

防止梯度爆炸：

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 然后再更新参数
optimizer.step()
```

```
梯度裁剪的作用：

原始梯度：[100, -200, 500]  ← 梯度太大！
裁剪后：  [0.2, -0.4, 1.0]  ← 按比例缩小

防止参数更新过大导致训练崩溃
```

---

## 8-bit 优化器

为了节省显存，可以使用量化优化器：

```python
import bitsandbytes as bnb

# 8-bit Adam（显存减少约 75%）
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5)

# 对比：
# 标准 Adam：优化器状态 64 GB
# 8-bit Adam：优化器状态 16 GB
```

---

## 常见问题

### Q: Loss 变成 NaN 怎么办？

```
可能原因：
1. 学习率太大
2. 梯度爆炸
3. 数据有问题

解决方法：
1. 减小学习率：1e-5 → 1e-6
2. 添加梯度裁剪：--max_grad_norm 1.0
3. 检查数据是否有 NaN/Inf
```

### Q: 训练很慢，Loss 下降很慢？

```
可能原因：
1. 学习率太小
2. Batch size 太小

解决方法：
1. 增大学习率：1e-6 → 1e-5
2. 增大 batch size 或梯度累积
```

### Q: 为什么大模型都用 AdamW？

```
原因：
1. 自适应学习率，不同参数不同步长
2. 动量加速收敛
3. 权重衰减防止过拟合
4. 对超参数不太敏感，容易调优
```

---

## 与训练参数的关系

| 参数 | 说明 |
|------|------|
| `--learning_rate` | 基础学习率 |
| `--weight_decay` | 权重衰减系数 |
| `--adam_beta1` | Adam 一阶矩系数 |
| `--adam_beta2` | Adam 二阶矩系数 |
| `--adam_epsilon` | Adam 数值稳定项 |
| `--max_grad_norm` | 梯度裁剪阈值 |
| `--optim` | 优化器类型 |

---

## 一句话总结

> **优化器**是模型的"教练"，Adam/AdamW 是最聪明的教练——它会根据每个参数的历史表现，给出个性化的学习建议，让训练又快又稳。

