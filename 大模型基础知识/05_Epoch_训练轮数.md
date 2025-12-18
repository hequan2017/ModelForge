# Epoch（训练轮数）

## 简单理解

**Epoch** 就是"遍数"，表示模型把整个训练数据集完整看了多少遍。1 个 Epoch = 所有数据都被模型学习了一次。

---

## 类比解释

```
假设你有一本 100 页的教材：

Epoch 1：从第 1 页读到第 100 页（第一遍）
         ┌────────────────────────┐
         │ 📖 读完整本书           │
         └────────────────────────┘

Epoch 2：从第 1 页再读到第 100 页（第二遍）
         ┌────────────────────────┐
         │ 📖 又读了一遍           │
         └────────────────────────┘

Epoch 3：再读一遍...
         ┌────────────────────────┐
         │ 📖 第三遍               │
         └────────────────────────┘

读的遍数越多，记得越牢（但可能死记硬背，不会举一反三）
```

---

## Epoch vs Step vs Iteration

```
假设：
- 数据集大小：1000 条
- Batch Size：100
- Epochs：3

计算：
- 1 个 Epoch = 1000 ÷ 100 = 10 Steps
- 总 Steps = 10 × 3 = 30 Steps

时间线：
├── Epoch 1 ──────────────────┤
│ Step 1 │ Step 2 │...│Step 10│
├── Epoch 2 ──────────────────┤
│Step 11 │Step 12 │...│Step 20│
├── Epoch 3 ──────────────────┤
│Step 21 │Step 22 │...│Step 30│
```

---

## Epoch 数量的影响

```
Epoch 太少（欠拟合）：
┌─────────────────────────────────────┐
│ 模型还没学会，Loss 还很高            │
│ 就像只读了一遍书就去考试              │
└─────────────────────────────────────┘

Epoch 合适：
┌─────────────────────────────────────┐
│ 模型学得刚刚好，泛化能力强            │
│ 就像认真复习了几遍，理解了知识         │
└─────────────────────────────────────┘

Epoch 太多（过拟合）：
┌─────────────────────────────────────┐
│ 模型死记硬背训练数据，不会举一反三     │
│ 就像把书背下来了，但换个题就不会       │
└─────────────────────────────────────┘
```

---

## 如何选择 Epoch 数量

| 数据量 | 推荐 Epoch | 说明 |
|--------|-----------|------|
| 大（>100k） | 1-3 | 数据多，少看几遍就够 |
| 中（10k-100k） | 3-5 | 适中 |
| 小（<10k） | 5-10 | 数据少，需要多看几遍 |
| 微调大模型 | 1-3 | 大模型已有知识，不需要太多 |

---

## 过拟合检测

```
Loss
  │
  │ ╲  训练 Loss（持续下降）
  │  ╲_______________
  │     ╱
  │    ╱  验证 Loss（开始上升）
  │   ╱
  │  ╱
  └──────────────────────→ Epoch
     1    2    3    4    5
              ↑
         最佳 Epoch（验证 Loss 最低点）
         
当验证 Loss 开始上升时，就应该停止训练！
这就是 "Early Stopping"（早停）
```

---

## 代码示例

```python
from transformers import TrainingArguments

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,           # 训练 3 个 Epoch
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",   # 每个 Epoch 结束时评估
    save_strategy="epoch",         # 每个 Epoch 结束时保存
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## 训练日志示例

```
Epoch 1/3
━━━━━━━━━━━━━━━━━━━━ 100% │ Step 10/10 │ Loss: 2.45 │ 

Epoch 2/3
━━━━━━━━━━━━━━━━━━━━ 100% │ Step 20/20 │ Loss: 1.82 │ 

Epoch 3/3
━━━━━━━━━━━━━━━━━━━━ 100% │ Step 30/30 │ Loss: 1.35 │ 

Training completed!
Best model saved at epoch 3 with loss 1.35
```

---

## 实际训练中的计算

```python
# 假设配置
dataset_size = 1000
per_device_batch_size = 4
num_gpus = 5
gradient_accumulation_steps = 4
num_epochs = 3

# 计算
effective_batch_size = per_device_batch_size * num_gpus * gradient_accumulation_steps
# = 4 * 5 * 4 = 80

steps_per_epoch = dataset_size // effective_batch_size
# = 1000 // 80 = 12

total_steps = steps_per_epoch * num_epochs
# = 12 * 3 = 36

print(f"每个 Epoch: {steps_per_epoch} 步")
print(f"总训练步数: {total_steps} 步")
```

---

## 与训练参数的关系

| 参数 | 与 Epoch 的关系 |
|------|----------------|
| `--num_train_epochs` | 直接设置训练轮数 |
| `--max_steps` | 设置最大步数（会覆盖 epoch 设置） |
| `--eval_strategy epoch` | 每个 epoch 结束时评估 |
| `--save_strategy epoch` | 每个 epoch 结束时保存 |
| `--logging_steps` | 每隔多少步记录日志 |

---

## 常见问题

### Q: Epoch 和 Step 该用哪个？

```
用 Epoch：
- 数据集大小固定
- 想要完整遍历数据集
- 适合小数据集

用 Step：
- 数据集很大或流式数据
- 想要精确控制训练量
- 适合大规模预训练
```

### Q: 如何判断 Epoch 够不够？

```
看 Loss 曲线：
- Loss 还在明显下降 → 增加 Epoch
- Loss 已经平稳 → Epoch 足够
- 验证 Loss 上升 → Epoch 太多，过拟合了
```

---

## 一句话总结

> **Epoch** 是模型学习数据的"遍数"——太少学不会，太多会死记硬背。通常大模型微调 1-3 个 Epoch 就够了。

