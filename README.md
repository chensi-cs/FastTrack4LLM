💡注：以下内容中，✅表示已经完成，❌表示有计划但尚未完成

<div align="center">

# 🚀 LLM Learning Project

**从零开始构建大语言模型的完整学习框架**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

</div>

## 🎯 项目愿景

在这个大模型时代，**LLM Learning** 是一个从零开始构建的大语言模型学习框架，致力于让每个人都能真正理解大语言模型的内部机制。不同于仅仅调用API或使用现成模型，我们带你从零开始，亲手构建、训练、优化属于自己的大语言模型。



本项目完整复现了大模型训练的全生命周期，让你深入理解每一个技术细节：

- 🏗️ **从零构建大模型结构**：基于Transformer-Decoder架构，支持多种开源模型结构的实现，支持MoE（混合专家模型）
- 🚀 **预训练 (Pretrain)**：大规模无监督预训练，让模型学会"词语接龙"
- 🎯 **监督微调 (SFT)**：指令微调和对齐训练，让模型学会"如何对话"
- ⚡ **LoRA微调**：参数高效微调技术，用少量参数实现大效果
- 🧠 **直接偏好优化 (DPO)**：基于人类反馈的强化学习，让模型更符合人类偏好
- 🔄 **模型蒸馏**：大模型知识迁移到小模型，实现"老师教学生"

致谢🙏：受开源项目 [MiniMind](https://github.com/jingyaogong/minimind) 的启发完成


## 📁 项目结构

```
llm_learning/
├── README.md                    # 项目说明文档
├── requirements.txt            # 依赖库列表
├── train.sh                    # 一键训练脚本
├── model_chat.py              # 交互式对话测试
├── checkpoints/               # 模型检查点保存目录
├── data/                      # 数据目录
│   ├── llm_data/              # 大模型训练数据
│   │   ├── raw/               # 原始数据
│   │   │   ├── pretrain_hq.jsonl    # 高质量预训练数据
│   │   │   ├── sft_mini_512.jsonl   # SFT微调数据
│   │   │   └── dpo.jsonl            # DPO偏好数据
│   │   └── processed/         # 处理后数据
│   ├── tokenizer.json         # 分词器配置
│   └── tokenizer_config.json  # 分词器参数
├── data_process/              # 数据处理工具
│   ├── data_convert.py        # JSONL转JSON
│   ├── data_print.py          # 数据查看工具
│   ├── parquet2csv.py         # 格式转换工具
│   └── split_data.py          # 数据切分工具
├── models/                    # 模型实现
│   ├── __init__.py
│   ├── attention.py           # 注意力机制实现
│   ├── feed_forward.py        # 前馈神经网络
│   ├── llama_model.py         # LLaMA模型结构
│   ├── norm_data.py           # 归一化层实现
│   ├── position_embed.py      # 位置编码实现
│   └── add_lora.py            # LoRA实现
├── trainner/                  # 训练框架
│   ├── __init__.py
│   ├── train_pretrain.py      # 预训练脚本
│   ├── train_sft.py           # SFT微调脚本
│   ├── train_lora.py          # LoRA微调脚本
│   ├── train_dpo.py           # DPO训练脚本
│   └── train_distillation.py  # 模型蒸馏脚本
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   ├── data.py                # 数据加载
│   └── utils.py               # 通用工具
├── scripts/                   # 训练脚本
│   ├── pretrain.sh            # 预训练启动脚本
│   ├── sft.sh                 # SFT微调脚本
│   ├── lora.sh                # LoRA微调脚本
│   ├── dpo.sh                 # DPO训练脚本
│   └── distill.sh             # 蒸馏训练脚本
├── logs/                      # 训练日志
├── wandb/                     # Weights & Biases实验跟踪
└── out/                       # 模型输出目录
```

## 🔧 环境准备

### 系统要求
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (推荐12.0+)
- **GPU**: 至少8GB显存 (推荐24GB显存)

### 快速安装
```bash
# 克隆项目
git clone https://github.com/chensi-cs/llm_learning
cd llm_learning

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证环境
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 📊 数据集准备

本项目不包括数据集的预处理过程，数据来源请参考开源项目 [MiniMind](https://github.com/jingyaogong/minimind) ，本项目使用的数据文件需放入`data/llm_data/processed/`目录下，具体的数据定义可参考`/utils/data.py`

| 数据集 | 大小 | 用途 |
|--------|------|------|
| pretrain_hq.jsonl | 1.6GB | 预训练 |
| sft_mini_512.jsonl | 1.2GB | 监督微调 | 
| dpo.jsonl | 909MB | 偏好优化 | 

### 数据格式示例

**预训练数据格式**：
```json
{"text": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器..."}
```

**SFT数据格式**：
```json
{
    "conversations": [
        {"role": "user", "content": "你好，请介绍一下自己"},
        {"role": "assistant", "content": "你好！我是一个AI助手，很高兴为你服务..."}
    ]
}
```

**DPO数据格式**：
```json
{
    "chosen": [
        {
            "content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?",
            "role": "user"
        },
        {
            "content": "To determine the number of moles of HBr required to react···",
            "role": "assistant"
        }
    ],
    "rejected": [
        {
            "content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?",
            "role": "user"
        },
        {
            "content": "To answer this question, we need to write down the chemical ···",
            "role": "assistant"
        }
    ]
}
```

## 🚀 训练流程详解

本项目支持完整的大模型训练生命周期：

### 1️⃣ 预训练 (Pretrain) ✅
**目标**：让模型学会"词语接龙"，建立基础的语言理解能力  

**输入`x`：** 每个样本都是一个tokens序列 `s` , 输入为`s[:-1]`，维度` (batch_size, sequence_length)`

**输出`pred`：** 维度` (batch_size, sequence_length,vocab_size)`，vocab_size 为词汇表大小

**损失计算：** 真实值 `y = s[1:]`（next token预测），用交叉熵损失函数`nn.CrossEntropyLoss`计算真实值 `y` 和预测值 `pred` 的损失

**启动命令**：
```bash
# 启动预训练脚本
bash scripts/pretrain.sh
```

**结果展示**：❌
### 2️⃣ 监督微调 (SFT) ✅
**目标**：让模型学会"如何对话"，理解指令和上下文  

**输入`x`：** 每个样本都是一个tokens序列 `s` , 输入为`s[:-1]`，维度` (batch_size, sequence_length)`

**输出`pred`：** 维度` (batch_size, sequence_length,vocab_size)`，vocab_size 为词汇表大小

**损失计算：** 真实值 `y = s[1:]`（next token预测），用交叉熵损失函数`nn.CrossEntropyLoss`计算真实值 `y` 和预测值 `pred` 的损失


**启动命令**：
```bash
# 快速开始
bash scripts/sft.sh

# 指定数据路径
python trainner/train_sft.py --data_path data/llm_data/processed/sft_mini_512.json
```
**结果展示**：❌

### 3️⃣ LoRA微调 ✅
**目标**：数高效微调，仅训练少量LoRA参数，冻结预训练权重，灵活配置超参数

**输入`x`：** 每个样本都是一个tokens序列 `s` , 输入为`s[:-1]`，维度` (batch_size, sequence_length)`

**输出`pred`：** 维度` (batch_size, sequence_length,vocab_size)`，vocab_size 为词汇表大小

**损失计算：** 真实值 `y = s[1:]`（next token预测），用交叉熵损失函数`nn.CrossEntropyLoss`计算真实值 `y` 和预测值 `pred` 的损失


**启动命令**：
```bash
# 基础LoRA训练
bash scripts/lora.sh

# 指定LoRA参数
python trainner/train_lora.py --data_path data/llm_data/processed/lora_medical.json --lora_rank 64 --lora_alpha 128
```
**结果展示**：❌

### 4️⃣ DPO偏好优化 ✅

**目标**：基于人类反馈优化模型回复质量  


**输入`x`：** DPO的一个样本有两个对话句子，一个是 `chosen` 句s1 = `prompt+good answer`,一个是 `reject` 句s2 = `prompt+bad answer`,他们的prompt都是相同的，模型的输入是 `torch.cat(s1[:-1],s2[:-1],dim=0)` , 其中 `s1` 和 `s2` 维度都是` (batch_size, sequence_length)`

**输出`pred`：** 维度` (batch_size, sequence_length,vocab_size)`，vocab_size 为词汇表大小，其中前一半 `batch_size` 是 `chosen` 句的输出，后一半 `batch_size` 是 `reject` 句的输出

**损失计算：** 真实值 `y = torch.cat(s1[1：],s2[1:],dim=0)`，损失的详细计算方式参考`/trainner/train_dpo.py`


**启动命令**：
```bash
# 启动DPO训练脚本
bash scripts/dpo.sh
```
**结果展示**：❌
### 5️⃣ 模型蒸馏 ✅
**目标**：大模型知识迁移到小模型
**启动命令**：
```bash
# 启动模型蒸馏脚本
bash scripts/distill.sh
```
**结果展示**：❌

## 模型结构

### 支持的模型架构
- **LLaMA系列模型**：实现LLaMA1，LLaMA2，LLaMA3模型结构 ✅
- **混合专家(MoE)**：支持混合专家(MoE)模型 （包含共享+独立专家）✅

### LLaMA系列模型
#### LLaMA1结构 ✅
LLaMA1 (2023) 结构Transformer 解码器架构做出了以下改进：
- **Pre-normalization + RMSNorm**：使用前置层归一化+RMSNorm归一化函数替换Transformer 解码器原有的Post-normalization + LayerNorm ✅
- **RoPE**: 采用了旋转位置嵌入（Rotary Positional Embedding, RoPE） ✅
- **SwiGLU 激活函数**: 使用SwiGLU激活函数 ✅

#### LLaMA2 & LLaMA3 结构 ✅

- LLaMA2 (2023)  在 LLaMA1的基础上将多头注意力机制（Multi-Head Attention）替换为分组查询注意力（Grouped Query Attention） ✅

- LLaMA3 (2024) 相比 LLaMA2 在模型结构上并没有太大变化，未采用 MoE（Mixture of Experts）架构，而依然采用 Dense FFN 结构 


## 🎯 模型测试与评估

### 交互式测试
```bash
# 启动对话测试
python model_chat.py
```
**结果展示**：❌

### 批量评估 ❌

## 📈 训练策略与监控

### 训练策略
- **学习率调度**：余弦退火 + 线性预热  ✅
- **梯度累积**：支持大批量训练  ✅
- **混合精度**：FP16训练加速  ✅
- **梯度裁剪**：防止梯度爆炸  ✅
- **Flash Attention**: 加速训练 ✅

### 支持的监控工具
- **TensorBoard**: 实时训练监控
- **Weights & Biases**: 实验跟踪和对比
- **自定义日志**: 详细的训练日志记录

### 启动监控
```bash
# TensorBoard
tensorboard --logdir=logs/

# Weights & Biases
wandb login
python trainner/train_pretrain.py --use_wandb
```

### 关键指标监控 
- **训练损失**: 每步实时更新 ✅
- **验证困惑度**: 每轮评估 ❌
- **GPU利用率**: 硬件资源监控 ❌
- **学习率**: 动态调度曲线 ❌
- **梯度范数**: 梯度稳定性监控 ❌



## 🎓 学习路径推荐

### 初学者路径
1. **环境搭建** → 安装依赖，验证GPU环境
2. **数据准备** → 下载并理解数据集格式
3. **预训练** → 运行预训练脚本，观察loss下降
4. **SFT微调** → 训练对话模型，测试效果
5. **LoRA实验** → 尝试不同任务的LoRA微调

### 进阶路径
1. **源码阅读** → 深入理解每个模块的实现
2. **架构改进** → 尝试修改模型结构
3. **训练优化** → 实验不同的训练策略
4. **评估指标** → 设计更全面的评估方案
5. **生产部署** → 将模型部署到实际应用

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献
- 🐛 **报告Bug**: 提交Issue描述问题
- 💡 **功能建议**: 提出新功能的想法
- 📖 **文档改进**: 完善README和代码注释
- 🔧 **代码贡献**: 提交Pull Request

### 开发计划
- [ ] 支持更多模型架构 (如Qwen)
- [ ] 增加中文分词器训练
- [ ] 完善模型评估指标
- [ ] 支持模型量化部署
- [ ] 增加Web界面Demo
- [ ] 多模态扩展 (图文理解)

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) - 项目灵感来源,提供技术参考
- [LLaMA](https://github.com/facebookresearch/llama) - 模型架构参考
- [Hugging Face](https://huggingface.co/) - 开源生态支持
- **PyTorch团队**: 深度学习框架

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star支持！**  
**🚀 让我们一起探索大语言模型的奥秘！**

</div>