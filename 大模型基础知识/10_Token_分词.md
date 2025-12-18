# Token 与分词（Tokenization）

## 简单理解

**Token** 是模型处理文本的最小单位。分词就是把文本切成一个个 Token，就像把句子切成一块块积木。

---

## 类比解释

```
人类阅读：一个字一个字读
"今天天气很好" → 今 | 天 | 天 | 气 | 很 | 好

模型阅读：一个 Token 一个 Token 读
"今天天气很好" → 今天 | 天气 | 很 | 好

Token 不一定是单个字，可能是词、子词、甚至字节
```

---

## 为什么需要分词？

```
计算机不认识文字，只认识数字：

文本: "你好世界"
      ↓ 分词
Tokens: ["你好", "世界"]
      ↓ 转换成 ID
Token IDs: [12345, 67890]
      ↓ 转换成向量
Embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
      ↓
模型可以处理了！
```

---

## 常见分词方法

### 1. 字符级（Character-level）

```
"Hello" → ['H', 'e', 'l', 'l', 'o']

优点：词表小，不会遇到未知词
缺点：序列太长，丢失词义信息
```

### 2. 词级（Word-level）

```
"Hello world" → ['Hello', 'world']

优点：保留完整词义
缺点：词表巨大，无法处理新词（OOV 问题）
```

### 3. 子词级（Subword-level）⭐ 主流方法

```
"unhappiness" → ['un', 'happiness'] 或 ['un', 'happ', 'iness']

优点：
- 词表大小适中（32K-150K）
- 能处理新词（拆成已知子词）
- 保留一定的语义信息
```

---

## BPE 算法（Byte Pair Encoding）

大多数大模型使用的分词算法：

```
训练过程：

初始词表：所有单个字符
['a', 'b', 'c', ..., 'z', ' ', ...]

Step 1: 统计相邻字符对的频率
"low lower lowest" 中：
'l'+'o' 出现 3 次
'o'+'w' 出现 3 次
'e'+'r' 出现 1 次
...

Step 2: 合并最高频的字符对
'l'+'o' → 'lo' (新 token)

Step 3: 重复直到达到目标词表大小

最终词表：
['a', ..., 'z', 'lo', 'low', 'er', 'est', ...]
```

---

## 实际分词示例

### 英文分词

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['Hello', ',', ' how', ' are', ' you', '?']

ids = tokenizer.encode(text)
print(ids)
# [9707, 11, 1246, 525, 498, 30]
```

### 中文分词

```python
text = "今天天气很好"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['今天', '天气', '很', '好']

# 中文大约 1 个字 = 1-2 个 token
```

### 代码分词

```python
text = "def hello_world():"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['def', ' hello', '_', 'world', '():', ]

# 代码中的符号和关键字通常是独立 token
```

---

## Token 数量估算

```
经验法则：

英文：1 个单词 ≈ 1-2 个 tokens
中文：1 个汉字 ≈ 1-2 个 tokens
代码：变化较大，符号多

例如：
"Hello world" = 2-3 tokens
"今天天气很好" = 4-6 tokens
"def func():" = 4-5 tokens

快速估算：
- 英文：字符数 ÷ 4
- 中文：字符数 × 1.5
```

---

## max_length 参数

```
--max_length 4096

含义：输入序列最多 4096 个 tokens

如果超过：
┌─────────────────────────────────────────────┐
│ 原文：[token1, token2, ..., token5000]       │
│                                             │
│ 截断后：[token1, token2, ..., token4096]     │
│                        后面的被丢弃 ↗        │
└─────────────────────────────────────────────┘

影响：
- 长文本会丢失信息
- 长度越大，显存占用越高（平方关系）
```

---

## 特殊 Token

```
模型需要一些特殊标记来理解输入结构：

[BOS] - Beginning of Sequence（序列开始）
[EOS] - End of Sequence（序列结束）
[PAD] - Padding（填充）
[UNK] - Unknown（未知词）
[SEP] - Separator（分隔符）

示例：
用户输入："你好"
实际输入：[BOS] 你好 [EOS]

对话格式：
[BOS] <|user|> 你好 <|assistant|> 你好！有什么可以帮你？ [EOS]
```

---

## 对话模板

不同模型有不同的对话格式：

### ChatML 格式（Qwen 使用）

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的吗？<|im_end|>
```

### Llama 格式

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

你好 [/INST] 你好！有什么可以帮你的吗？ </s>
```

---

## 代码示例

```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 基本分词
text = "人工智能正在改变世界"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"Token 数量: {len(tokens)}")

# 编码（文本 → ID）
input_ids = tokenizer.encode(text)
print(f"Token IDs: {input_ids}")

# 解码（ID → 文本）
decoded = tokenizer.decode(input_ids)
print(f"解码结果: {decoded}")

# 批量处理
texts = ["你好", "世界"]
encoded = tokenizer(texts, padding=True, return_tensors="pt")
print(f"Input IDs shape: {encoded['input_ids'].shape}")

# 查看词表大小
print(f"词表大小: {tokenizer.vocab_size}")
```

---

## 分词对训练的影响

### 1. 序列长度

```
同样的文本，不同分词器得到的长度不同：

GPT-2 分词器："今天天气很好" → 6 tokens
Qwen 分词器："今天天气很好" → 4 tokens

Qwen 对中文更友好，同样长度能处理更多内容
```

### 2. 显存占用

```
显存 ∝ (序列长度)²

max_length = 2048: 基准显存
max_length = 4096: 约 4 倍显存
max_length = 8192: 约 16 倍显存
```

### 3. 训练效率

```
Token 数量影响训练时间：

数据集 1M 条，平均 500 tokens/条
总 Token 数 = 1M × 500 = 500M tokens

训练速度 1000 tokens/秒
训练时间 = 500M ÷ 1000 = 500000 秒 ≈ 139 小时
```

---

## 常见问题

### Q: 为什么有时候输出会断在奇怪的地方？

```
因为模型是按 token 生成的，可能在子词中间停止：

生成中：["今", "天", "天气", "很"]
下一个 token 概率：
  "好" = 0.3
  "棒" = 0.25
  [EOS] = 0.35  ← 如果选中这个，就停止了

解决：调整生成参数（temperature, top_p 等）
```

### Q: 为什么中文模型效果更好？

```
中文优化的分词器：
- 词表包含更多中文词汇
- 常用词作为整体 token（如 "今天"、"天气"）
- 减少 token 数量，提高效率

通用分词器可能把 "今天" 拆成 "今" + "天"
```

---

## 与训练参数的关系

| 参数 | 与 Token 的关系 |
|------|----------------|
| `--max_length` | 最大 token 数量 |
| `--per_device_train_batch_size` | 每批处理多少条数据 |
| 显存占用 | 与 token 数量的平方成正比 |
| 训练速度 | 与总 token 数量成正比 |

---

## 一句话总结

> **Token** 是模型的"视觉单位"，就像人看字一样，模型看 Token。分词决定了模型如何"阅读"文本，好的分词器能让模型更高效地理解语言。

