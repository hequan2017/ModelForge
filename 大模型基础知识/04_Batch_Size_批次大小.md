# Batch Size（批次大小）

## 简单理解

**Batch Size** 就是模型每次"看"多少条数据后才更新一次参数。就像老师批改作业，可以改一份就讲评，也可以改完一批再统一讲评。

---

## 类比解释

```
Batch Size = 1（随机梯度下降 SGD）
┌─────┐
│作业1│ → 讲评 → 调整教学
└─────┘
┌─────┐
│作业2│ → 讲评 → 调整教学
└─────┘
┌─────┐
│作业3│ → 讲评 → 调整教学
└─────┘
优点：更新频繁
缺点：方向不稳定（一份作业不能代表全班水平）


Batch Size = 3（小批量梯度下降 Mini-batch SGD）
┌─────┬─────┬─────┐
│作业1│作业2│作业3│ → 统一讲评 → 调整教学
└─────┴─────┴─────┘
优点：方向更准确
缺点：需要更多内存存放这批作业
```

---

## Batch Size 的影响

| Batch Size | 训练稳定性 | 收敛速度 | 显存占用 | 泛化能力 |
|------------|-----------|---------|---------|---------|
| 小 (1-8) | 不稳定 | 慢 | 低 | 可能更好 |
| 中 (16-64) | 较稳定 | 适中 | 中 | 较好 |
| 大 (128+) | 很稳定 | 快 | 高 | 可能变差 |

---

## 有效 Batch Size 计算

在分布式训练中，实际的 batch size 需要考虑多个因素：

```
有效 Batch Size = per_device_batch_size × GPU数量 × gradient_accumulation_steps

例如：
- per_device_train_batch_size = 4
- GPU 数量 = 5
- gradient_accumulation_steps = 4

有效 Batch Size = 4 × 5 × 4 = 80
```

---

## 图解：梯度累积

当显存不够用大 batch 时，可以用梯度累积模拟：

```
直接用 Batch Size = 8（显存不够）：
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ → 计算梯度 → 更新参数
└───┴───┴───┴───┴───┴───┴───┴───┘
                                    显存爆炸！❌


梯度累积 = 4，Batch Size = 2（显存够用）：
┌───┬───┐
│ 1 │ 2 │ → 计算梯度 → 累积
└───┴───┘
┌───┬───┐
│ 3 │ 4 │ → 计算梯度 → 累积
└───┴───┘
┌───┬───┐
│ 5 │ 6 │ → 计算梯度 → 累积
└───┴───┘
┌───┬───┐
│ 7 │ 8 │ → 计算梯度 → 累积 → 更新参数
└───┴───┘
                                    效果等同于 Batch Size = 8 ✅
```

---

## 显存占用估算

```
显存占用 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值

激活值 ∝ Batch Size × 序列长度²

例如 8B 模型：
- 模型参数：16GB
- 梯度：16GB
- 优化器状态：32GB（Adam 需要 2 倍）
- 激活值：Batch Size × 序列长度² × 常数

Batch Size = 1, 长度 = 4096：激活值 ≈ 10GB
Batch Size = 4, 长度 = 4096：激活值 ≈ 40GB
```

---

## 代码示例

```python
from torch.utils.data import DataLoader

# 创建数据加载器
train_loader = DataLoader(
    dataset,
    batch_size=4,           # 每批 4 条数据
    shuffle=True,           # 打乱顺序
    num_workers=4,          # 4 个进程并行加载
    drop_last=True          # 丢弃最后不完整的批次
)

# 训练循环
for batch in train_loader:
    inputs, labels = batch
    print(f"Batch shape: {inputs.shape}")  # [4, seq_len]
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    loss.backward()
    
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
```

---

## 梯度累积代码示例

```python
gradient_accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(train_loader):
    inputs, labels = batch
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 缩放 loss（因为要累积多次）
    loss = loss / gradient_accumulation_steps
    
    # 反向传播（累积梯度）
    loss.backward()
    
    # 每累积 4 步才更新一次
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Batch Size 调优建议

### 显存不足时

```bash
# 原配置（OOM）
--per_device_train_batch_size 8
--gradient_accumulation_steps 1

# 优化后（显存减半，效果相同）
--per_device_train_batch_size 4
--gradient_accumulation_steps 2

# 或者更激进
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

### 训练不稳定时

```bash
# 增大有效 batch size
--per_device_train_batch_size 2
--gradient_accumulation_steps 16  # 有效 batch = 2 × 5 × 16 = 160
```

### 想要加速训练

```bash
# 如果显存充足，增大 batch size
--per_device_train_batch_size 8
--gradient_accumulation_steps 2

# 同时可以适当增大学习率
--learning_rate 2e-5
```

---

## 与训练参数的关系

| 参数 | 说明 |
|------|------|
| `--per_device_train_batch_size` | 每个 GPU 的批次大小 |
| `--per_device_eval_batch_size` | 评估时的批次大小 |
| `--gradient_accumulation_steps` | 梯度累积步数 |
| `--dataloader_num_workers` | 数据加载并行数 |

---

## 一句话总结

> **Batch Size** 决定了模型每次学习多少样本——太小方向不准，太大显存爆炸。用梯度累积可以在有限显存下模拟大 batch 的效果。

