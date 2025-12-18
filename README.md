# ModelForge
大模型训练测试

# Qwen3-30B-A3B 全参数微调完整流程指南

## 📋 流程概览

```
环境准备 → 数据准备 → 配置检查 → 开始训练 → 监控训练 → 模型合并 → 模型测试 → 模型部署
```

---

## 🔧 第一步：环境准备

### 1.1 安装 CUDA 和 PyTorch

```bash
# 检查 CUDA 版本
nvidia-smi
# 输出中会显示 CUDA Version，确保 >= 11.8
```

### 1.2 安装 ms-swift（魔搭训练框架）

```bash
# 创建虚拟环境（推荐）
conda create -n swift python=3.10 -y
conda activate swift

# 安装 swift
pip install 'ms-swift[llm]' -U

# 安装 deepspeed（分布式训练必需）
pip install deepspeed -U

# 验证安装
swift --version
```
**命令含义**：
- `conda create`：创建独立的Python环境，避免包冲突
- `pip install 'ms-swift[llm]'`：安装swift及其LLM相关依赖
- `-U`：升级到最新版本

### 1.3 检查 GPU 状态

```bash
# 查看所有GPU
nvidia-smi

# 检查PyTorch能否识别GPU
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

---

## 📊 第二步：数据准备

### 2.1 使用在线数据集（本脚本方式）

```bash
# 脚本中使用的是HuggingFace数据集，会自动下载
--dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k#1000'
```
**格式说明**：
- `liucong/Chinese-DeepSeek-R1-Distill-data-110k`：数据集路径
- `#1000`：只取前1000条，用于测试

### 2.2 使用本地数据集（可选）

如果你有自己的数据，需要准备成以下格式：

**方式一：JSONL格式**（推荐）
```json
{"messages": [{"role": "system", "content": "你是一个助手"}, {"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
{"messages": [{"role": "user", "content": "问题2"}, {"role": "assistant", "content": "回答2"}]}
```

**方式二：JSON格式**
```json
[
  {"instruction": "问题1", "output": "回答1"},
  {"instruction": "问题2", "input": "补充信息", "output": "回答2"}
]
```

保存为 `train.jsonl`，然后修改脚本：
```bash
--dataset train.jsonl
```

### 2.3 验证数据格式

```bash
# 查看数据前几行
head -n 3 train.jsonl

# 验证JSON格式是否正确
python -c "import json; [json.loads(l) for l in open('train.jsonl')][:3]"
```

---

## ✅ 第三步：配置检查

### 3.1 检查脚本参数

```bash
# 查看脚本内容
cat train_qwen3_lora.sh
```

### 3.2 关键参数确认清单

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| GPU数量 | `nvidia-smi --list-gpus` | 5张GPU |
| 磁盘空间 | `df -h .` | 至少200GB可用 |
| 内存 | `free -h` | 建议64GB以上 |

### 3.3 测试模型下载（可选）

```bash
# 先单独下载模型，避免训练时下载超时
swift download --model Qwen/Qwen3-30B-A3B-Thinking-2507
```
**命令含义**：预先下载模型到本地缓存（~/.cache/modelscope 或 ~/.cache/huggingface）

---

## 🚀 第四步：开始训练

### 4.1 赋予执行权限（Linux/WSL）

```bash
chmod +x train_qwen3_lora.sh
```
**命令含义**：给脚本添加可执行权限

### 4.2 启动训练

```bash
# 方式一：直接运行
bash train_qwen3_lora.sh

# 方式二：后台运行（推荐，断开SSH也不会停止）
nohup bash train_qwen3_lora.sh > train.log 2>&1 &
```
**命令含义**：
- `nohup`：忽略挂断信号，断开终端也继续运行
- `> train.log`：标准输出重定向到日志文件
- `2>&1`：错误输出也写入同一文件
- `&`：后台运行

### 4.3 查看训练日志

```bash
# 实时查看日志
tail -f train.log

# 查看最后100行
tail -n 100 train.log

# 搜索错误信息
grep -i "error\|exception" train.log
```

---

## 📈 第五步：监控训练

### 5.1 监控 GPU 使用

```bash
# 实时监控GPU（每1秒刷新）
watch -n 1 nvidia-smi

# 或使用更友好的工具
pip install gpustat
gpustat -i 1
```

### 5.2 查看训练进度

训练日志中会显示：
```
{'loss': 2.345, 'learning_rate': 1e-05, 'epoch': 0.5, 'global_step': 20}
```
- `loss`：损失值，应该逐渐下降
- `epoch`：当前轮次进度
- `global_step`：当前训练步数

### 5.3 使用 TensorBoard 可视化（可选）

```bash
# 安装
pip install tensorboard

# 启动（在output目录）
tensorboard --logdir output_qwen3_30b_full --port 6006

# 浏览器访问 http://localhost:6006
```

---

## 🔗 第六步：模型合并（如果使用LoRA）

> ⚠️ 本脚本使用全参数微调（full），**不需要合并**，可跳过此步骤

如果使用 LoRA 微调，需要将 adapter 合并到基座模型：

```bash
swift merge \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --adapters output_qwen3_30b_full/checkpoint-xxx \
  --output_dir merged_model
```
**命令含义**：
- `--adapters`：LoRA权重路径
- `--output_dir`：合并后模型保存位置

---

## 🧪 第七步：模型测试

### 7.1 命令行测试

```bash
swift infer \
  --model output_qwen3_30b_full/checkpoint-best \
  --stream true
```
**命令含义**：
- `swift infer`：启动推理模式
- `--stream true`：流式输出，打字机效果

### 7.2 交互式对话测试

```bash
swift app \
  --model output_qwen3_30b_full/checkpoint-best \
  --port 7860
```
**命令含义**：启动 Gradio Web界面，浏览器访问 http://localhost:7860

### 7.3 批量测试

```bash
# 准备测试数据 test.jsonl
{"query": "什么是人工智能？"}
{"query": "解释一下量子计算"}

# 运行批量推理
swift infer \
  --model output_qwen3_30b_full/checkpoint-best \
  --val_dataset test.jsonl \
  --result_path results.jsonl
```

---

## 🌐 第八步：模型部署

### 8.1 部署为 API 服务

```bash
swift deploy \
  --model output_qwen3_30b_full/checkpoint-best \
  --port 8000
```
**命令含义**：启动 OpenAI 兼容的 API 服务

### 8.2 测试 API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 512
  }'
```

### 8.3 使用 vLLM 高性能部署（推荐生产环境）

```bash
# 安装 vLLM
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model output_qwen3_30b_full/checkpoint-best \
  --tensor-parallel-size 5 \
  --port 8000
```
**命令含义**：
- `--tensor-parallel-size 5`：使用5张GPU进行张量并行
- vLLM 提供更高的推理吞吐量

---

## 🔥 常见问题处理

### Q1: CUDA Out of Memory（显存不足）

```bash
# 解决方案1：减小批次大小
--per_device_train_batch_size 1

# 解决方案2：减小序列长度
--max_length 2048

# 解决方案3：使用更强的显存优化
--deepspeed zero3_offload
```

### Q2: 训练速度太慢

```bash
# 增加数据加载workers
--dataloader_num_workers 8

# 如果显存允许，增加批次大小
--per_device_train_batch_size 2
```

### Q3: 模型下载失败

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或使用魔搭
export USE_MODELSCOPE=1
```

### Q4: 如何恢复中断的训练

```bash
# 添加 resume 参数，指向最后一个checkpoint
--resume_from_checkpoint output_qwen3_30b_full/checkpoint-50
```

---

## 📁 输出文件说明

训练完成后，`output_qwen3_30b_full/` 目录结构：

```
output_qwen3_30b_full/
├── checkpoint-50/          # 第50步的检查点
│   ├── config.json         # 模型配置
│   ├── model.safetensors   # 模型权重
│   └── tokenizer.json      # 分词器
├── checkpoint-75/          # 最终检查点
├── runs/                   # TensorBoard日志
├── training_args.json      # 训练参数记录
└── trainer_state.json      # 训练状态
```

---

## ✅ 快速检查清单

开始训练前，确认以下事项：

- [ ] CUDA和PyTorch已安装
- [ ] ms-swift已安装：`swift --version`
- [ ] GPU可用：`nvidia-smi`
- [ ] 磁盘空间充足（>200GB）
- [ ] 数据集路径正确
- [ ] 脚本参数已确认

---

祝训练顺利！🎉

