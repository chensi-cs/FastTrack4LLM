💡注：以下内容中，✅表示已经完成，❌表示有计划但尚未完成
# LLM Learning Project

本项目是一个从零开始构建的大语言模型学习框架，受开源项目 [MiniMind](https://github.com/jingyaogong/minimind) 的启发，实现了完整的大模型训练，微调，推理流程。

## 🎯 项目特色

本项目完整复现了大模型训练的全生命周期，包括：
- 🏗️ **从零构建大模型结构**：基于Transformer-Decoder架构，支持MoE（混合专家模型）✅
- 🚀 **预训练 (Pretrain)**：大规模无监督预训练 ✅
- 🎯 **监督微调 (SFT)**：指令微调和对齐训练 ✅
- ⚡ **LoRA微调**：参数高效微调技术 ✅
- 🧠 **直接偏好优化 (DPO)**：基于人类反馈的强化学习 ✅
- 🔄 **模型蒸馏**：大模型知识迁移到小模型 ✅

## 📁 项目结构

```
llm_learning/
├── README.md                    # 项目说明
├── requirements.txt            # 依赖库
├── train.sh                    # 训练启动脚本
├── model_chat.py              # 模型推理测试
├── checkpoints/               # 模型检查点保存目录
├── data/                      # 数据目录
│   ├── llm_data/              # 大模型训练数据
│   │   ├── raw/               # 原始数据
│   │   │   ├── pretrain_hq.jsonl    # 高质量预训练数据
│   │   │   ├── sft_mini_512.jsonl   # SFT微调数据
│   │   │   └── dpo.jsonl            # DPO偏好数据
│   │   └── processed/         # 处理后数据
│   │       ├── pretrain_hq.json     # 预训练JSON格式
│   │       ├── sft_mini_512.json    # SFT数据JSON格式
│   │       └── dpo.json             # DPO数据JSON格式
│   ├── tokenizer.json         # 分词器配置
│   └── tokenizer_config.json  # 分词器参数
├── data_process/              # 数据处理工具
│   ├── data_convert.py        # JSONL转JSON
│   ├── data_print.py          # 数据查看工具
│   ├── parquet2csv.py         # 格式转换工具
│   └── split_data.py          # 数据切分工具
├── models/                    # 模型实现
│   ├── __init__.py
│   ├── attention.py           # 注意力机制
│   ├── feed_forward.py        # 前馈神经网络
│   ├── llama_model.py         # LLaMA模型
│   ├── norm_data.py           # 归一化层
│   ├── position_embed.py      # 位置编码
│   ├── add_lora.py            # LoRA实现
├── trainner/                  # 训练框架
│   ├── __init__.py
│   ├── train_pretrain.py      # 预训练脚本
│   ├── train_sft.py           # SFT微调脚本
│   ├── train_lora.py          # LoRA微调脚本
│   ├── train_dpo.py           # DPO训练脚本
│   ├── train_distillation.py  # 模型蒸馏脚本
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   ├── data.py                # 数据加载
│   └── utils.py               # 通用工具
├── scripts/                   # 训练脚本
│   ├── lora.sh          # LoRA训练脚本
│   ├── dpo.sh           # DPO训练脚本
│   ├── distill.sh       # 蒸馏训练脚本
│   ├── sft.sh           # SFT训练脚本
│   └── pretrain.sh            # 预训练脚本

```


## 🔧 核心依赖
```bash
torch>=2.0.0          # 深度学习框架
transformers>=4.30.0  # Hugging Face生态
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

本项目不包括数据集的预处理过程，数据来源请参考开源项目 [MiniMind](https://github.com/jingyaogong/minimind) ，本项目使用的数据文件需放入`data/llm_data/processed/`目录下

### 3. 训练流程

本项目支持完整的大模型训练生命周期，按以下顺序执行：

#### 1）预训练 ✅
**技术特点**：大规模无监督学习，因果语言建模，支持混合精度训练、梯度累积
```bash
# 启动预训练脚本
bash scripts/pretrain.sh
```

#### 2）监督微调 (SFT) ✅
**技术特点**：指令跟随训练，参数全量更新，支持多轮对话数据，早停机制防止过拟合
```bash
# 启动微调脚本
bash scripts/sft.sh
```

#### 3）LoRA微调 ✅
**技术特点**：参数高效微调，仅训练少量LoRA参数，冻结预训练权重，灵活配置超参数
```bash
# 启动LoRA微调脚本
bash scripts/lora.sh
```

#### 4）DPO偏好优化 ✅
**技术特点**：直接偏好优化，基于人类反馈的强化学习，使用成对偏好数据训练
```bash
# 启动DPO训练脚本
bash scripts/dpo.sh
```

#### 5）模型蒸馏 ✅
**技术特点**：知识迁移，软硬标签结合，温度调节软标签分布，大模型知识迁移到小模型
```bash
# 启动模型蒸馏脚本
bash scripts/distill.sh
```

### 4. 模型测试

```bash
# 交互式对话测试
python model_chat.py 
```

## 5. 训练细节

### 日志和可视化
- **TensorBoard**：`logs/` 目录下的训练日志 ✅
- **Weights & Biases**：`wandb/` 目录下的实验跟踪 ✅
- **训练图表**：每个epoch的损失曲线自动保存 ✅

### 性能指标
- **训练损失**：实时显示在TensorBoard ✅
- **验证困惑度**：每轮评估模型性能 ❌
- **GPU利用率**：监控硬件资源使用 ❌

### 训练策略
- **学习率调度**：余弦退火 + 线性预热  ✅
- **梯度累积**：支持大批量训练  ✅
- **混合精度**：FP16训练加速  ✅
- **梯度裁剪**：防止梯度爆炸  ✅
- **Flash Attention**: 加速训练 ✅

## 🔧 核心实现

### 1. 支持的模型架构
- **LLaMA系列模型**：实现LLaMA1，LLaMA2，LLaMA3模型结构 ✅
- **混合专家(MoE)**：支持混合专家(MoE)模型 （包含共享+独立专家）✅

### 2.LLaMA系列模型
#### 1）LLaMA1结构 ✅
LLaMA1 (2023) 结构Transformer 解码器架构做出了以下改进：
- **Pre-normalization + RMSNorm**：使用前置层归一化+RMSNorm归一化函数替换Transformer 解码器原有的Post-normalization + LayerNorm ✅
- **RoPE**: 采用了旋转位置嵌入（Rotary Positional Embedding, RoPE） ✅
- **SwiGLU 激活函数**: 使用SwiGLU激活函数 ✅

#### 2）LLaMA2 & LLaMA3 结构 ✅

- LLaMA2 (2023)  在 LLaMA1的基础上将多头注意力机制（Multi-Head Attention）替换为分组查询注意力（Grouped Query Attention） ✅

- LLaMA3 (2024) 相比 LLaMA2 在模型结构上并没有太大变化，未采用 MoE（Mixture of Experts）架构，而依然采用 Dense FFN 结构 

## 🎯 实验结果 ❌

### 预训练效果 ❌

### 微调效果 ❌

### 模型测试 ❌

## 🤝 贡献指南

欢迎提交Issue和PR！当前重点开发：
- [ ] 优化tokenizer
- [ ] Web界面部署
- [ ] 模型压缩与量化
- [ ] 多GPU分布式训练
- [ ] 模型评估指标完善

## 🙏 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) - 项目灵感来源
- [LLaMA](https://github.com/facebookresearch/llama) - 模型架构参考
- [Hugging Face](https://huggingface.co/) - 开源生态支持

---

**⭐ 如果这个项目对你有帮助，请给个Star支持！**