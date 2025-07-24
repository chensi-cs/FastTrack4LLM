# LLM Learning Project

本项目是一个基于Transformer架构的深度学习项目，主要用于中英文翻译任务和语言模型预训练。以下是项目结构和各模块的详细说明。

## 项目结构

```
llm_learning/
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖库列表
├── checkpoints/             # 模型检查点保存目录
│   └── best_model.pth       # 最佳模型权重文件
├── data/                    # 数据目录
│   ├── llm_data/            # 语言模型数据
│   │   ├── processed/       # 处理后的数据
│   │   └── raw/             # 原始数据
│   └── translate_data/      # 翻译任务数据
│       ├── processed/       # 处理后的翻译数据
│       └── raw/             # 原始翻译数据
├── models/                  # 模型实现
│   ├── attention.py         # 注意力机制实现
│   ├── llama_model.py       # LLaMA模型实现
│   └── transformer_model.py # Transformer模型实现
├── trainner/                # 训练脚本
│   ├── train_pretrain.py    # 预训练脚本
│   └── train_transformer.py # Transformer训练脚本
├── utils/                   # 工具模块
│   ├── activations.py       # 激活函数
│   ├── attention.py         # 注意力工具
│   ├── config.py           # 配置文件
│   ├── data.py             # 数据加载工具
│   ├── data_tokenization.py # 数据分词工具
│   ├── feed_forward.py      # 前馈网络
│   ├── norm_data.py        # 归一化工具
│   ├── optimizer.py        # 优化器
│   └── position_embed.py   # 位置编码
└── test/                    # 测试脚本
    ├── test.py             # 主测试脚本
    └── test_tokenizer.py  # 分词器测试
```

## 模块功能说明

### 1. 数据模块 (`data/`)
- **原始数据**：存放未经处理的原始数据文件（JSONL/CSV格式）。
- **处理后的数据**：经过清洗、分词和格式转换后的数据，可直接用于模型训练。

### 2. 模型模块 (`models/`)
- **Transformer模型**：基于标准的Transformer架构实现。
- **LLaMA模型**：实现了LLaMA架构的轻量级语言模型。
- **注意力机制**：支持多头注意力和旋转位置编码。

### 3. 训练模块 (`trainner/`)
- **预训练脚本**：用于语言模型的预训练。
- **翻译任务脚本**：支持中英文翻译任务的微调。

### 4. 工具模块 (`utils/`)
- **数据加载**：支持从JSON/CSV文件加载数据。
- **分词工具**：包含BPE分词器和自定义分词逻辑。
- **优化器**：实现了AdamW和带学习率热身的优化器。

## 支持的模型架构

### 1. LLaMA (Meta)
- **LLaMA1**：轻量级开源语言模型，支持7B/13B/33B/65B参数版本。

### 待补充


## 模型扩展计划
- [✅] 支持LLaMA1
- [❌] 支持LLaMA2/Llama3
- [❌] 支持Qwen
- [❌] 支持GPT系列

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行预训练：
   ```bash
   python trainner/train_pretrain.py
   ```



## TODO
- [ ] 增加更多主流模型结构（如GPT、BERT）。
- [ ] 优化数据加载速度。