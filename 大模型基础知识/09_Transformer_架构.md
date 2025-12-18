# Transformer 架构

## 简单理解

**Transformer** 是现代大语言模型的基础架构，它的核心思想是"注意力机制"——让模型能够关注输入中最重要的部分。

---

## 类比解释

```
传统方法（RNN）：像读书一样，一个字一个字往后读
┌─────────────────────────────────────┐
│ "今天天气很好" → 今→天→天→气→很→好   │
│                                     │
│ 问题：读到后面就忘了前面              │
└─────────────────────────────────────┘

Transformer：像考试一样，可以来回看整篇文章
┌─────────────────────────────────────┐
│ "今天天气很好"                       │
│   ↑↓  ↑↓  ↑↓                        │
│ 每个字都能看到其他所有字              │
│                                     │
│ 优点：能捕捉长距离依赖关系            │
└─────────────────────────────────────┘
```

---

## 整体架构

```
输入文本: "今天天气"
    │
    ▼
┌─────────────────────┐
│   Token Embedding   │  ← 把文字转成向量
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Position Embedding  │  ← 加入位置信息
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│                     │
│   Transformer       │
│   Layer × N         │  ← 堆叠 N 层（如 32 层）
│                     │
│  ┌───────────────┐  │
│  │ Self-Attention │  │  ← 注意力机制
│  └───────────────┘  │
│         │           │
│  ┌───────────────┐  │
│  │  Feed Forward  │  │  ← 前馈网络
│  └───────────────┘  │
│                     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│    Output Layer     │  ← 输出预测
└─────────────────────┘
    │
    ▼
预测下一个词: "很"
```

---

## 核心组件详解

### 1. Token Embedding（词嵌入）

```
把文字转换成计算机能理解的向量：

"今" → [0.2, -0.5, 0.8, ..., 0.1]  (维度如 4096)
"天" → [0.3, 0.1, -0.2, ..., 0.4]
"天" → [0.3, 0.1, -0.2, ..., 0.4]
"气" → [-0.1, 0.6, 0.3, ..., -0.2]

词表大小 × 嵌入维度 = 参数量
如：150000 × 4096 = 614M 参数
```

### 2. Position Embedding（位置编码）

```
告诉模型每个词的位置：

位置 0: "今" + [位置向量0]
位置 1: "天" + [位置向量1]
位置 2: "天" + [位置向量2]  ← 虽然是同一个字，但位置不同
位置 3: "气" + [位置向量3]

没有位置编码，模型分不清 "我爱你" 和 "你爱我"
```

### 3. Self-Attention（自注意力）

这是 Transformer 的核心！

```
问题：预测 "今天天气很___" 的下一个词

Self-Attention 的工作：
┌─────────────────────────────────────────────┐
│ "好" 应该关注哪些词？                         │
│                                             │
│  今   天   天   气   很                       │
│  ↑    ↑    ↑    ↑    ↑                       │
│  0.1  0.1  0.3  0.4  0.1  ← 注意力权重        │
│                                             │
│ "天气" 权重最高，因为和 "好" 最相关           │
└─────────────────────────────────────────────┘
```

#### Q、K、V 三兄弟

```
每个词都会生成三个向量：

Q (Query)：我在找什么？
K (Key)：  我有什么特征？
V (Value)：我的实际内容是什么？

计算过程：
1. Q × K^T = 注意力分数（谁和谁相关）
2. Softmax(分数) = 注意力权重（归一化）
3. 权重 × V = 加权输出（融合相关信息）

公式：Attention(Q,K,V) = Softmax(QK^T / √d) × V
```

### 4. Multi-Head Attention（多头注意力）

```
一个注意力头可能只关注一种关系，多个头可以关注多种关系：

Head 1：关注语法关系（主谓宾）
Head 2：关注语义关系（近义词）
Head 3：关注位置关系（相邻词）
...
Head 32：关注其他模式

最后把所有头的结果拼接起来
```

### 5. Feed Forward Network（前馈网络）

```
注意力层之后的"思考"层：

输入 → Linear(放大) → 激活函数 → Linear(缩小) → 输出

维度变化：4096 → 11008 → 4096

作用：增加模型的非线性表达能力
```

---

## 参数量计算

以 Qwen3-8B 为例：

```
模型配置：
- 层数 (num_layers): 32
- 隐藏维度 (hidden_size): 4096
- 注意力头数 (num_heads): 32
- FFN 维度 (intermediate_size): 11008
- 词表大小 (vocab_size): 151936

参数计算：
1. Embedding: 151936 × 4096 = 622M

2. 每层 Attention:
   - Q: 4096 × 4096 = 16.7M
   - K: 4096 × 4096 = 16.7M
   - V: 4096 × 4096 = 16.7M
   - O: 4096 × 4096 = 16.7M
   小计: 67M × 32层 = 2.1B

3. 每层 FFN:
   - Up: 4096 × 11008 = 45M
   - Down: 11008 × 4096 = 45M
   - Gate: 4096 × 11008 = 45M
   小计: 135M × 32层 = 4.3B

4. 其他 (LayerNorm, LM Head): ~1B

总计: 0.6B + 2.1B + 4.3B + 1B ≈ 8B 参数
```

---

## 推理过程

```
输入: "今天天气"
目标: 预测下一个词

Step 1: 编码
"今天天气" → [[向量1], [向量2], [向量3], [向量4]]

Step 2: 通过 32 层 Transformer
每一层都做：
  - Self-Attention（看看其他词）
  - FFN（思考处理）

Step 3: 输出层
最后一个位置的向量 → 预测所有词的概率
[0.01, 0.02, ..., 0.15, ..., 0.001]
                    ↑
                  "很" 概率最高

Step 4: 采样
选择 "很" 作为输出

Step 5: 继续预测
"今天天气很" → 预测下一个词 "好"
```

---

## 不同模型的架构对比

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|---------|---------|--------|
| GPT-2 | 12 | 768 | 12 | 117M |
| GPT-3 | 96 | 12288 | 96 | 175B |
| LLaMA-7B | 32 | 4096 | 32 | 7B |
| Qwen3-8B | 32 | 4096 | 32 | 8B |
| Qwen3-72B | 80 | 8192 | 64 | 72B |

---

## MoE（混合专家）架构

Qwen3-30B-A3B 使用的是 MoE 架构：

```
普通 FFN：
输入 → [FFN] → 输出

MoE FFN：
输入 → [路由器] → 选择专家
         │
    ┌────┼────┬────┬────┐
    ↓    ↓    ↓    ↓    ↓
  [专家1][专家2][专家3][专家4]...
    │         │
    └────+────┘  ← 只激活 2 个专家
         │
       输出

优点：
- 总参数 30B，但每次只用 3B（激活参数）
- 更大的模型容量，更少的计算量
```

---

## 代码示例

```python
import torch
import torch.nn as nn

class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)

# 使用示例
attention = SimpleSelfAttention(hidden_size=4096, num_heads=32)
x = torch.randn(2, 10, 4096)  # batch=2, seq_len=10
output = attention(x)
print(output.shape)  # [2, 10, 4096]
```

---

## 与训练的关系

| 组件 | 训练时的作用 |
|------|-------------|
| Embedding | 学习词的语义表示 |
| Attention | 学习词之间的关系 |
| FFN | 学习知识和推理能力 |
| LayerNorm | 稳定训练过程 |

---

## 一句话总结

> **Transformer** 是大模型的"骨架"，通过注意力机制让每个词都能"看到"其他所有词，从而理解语言的深层含义。

