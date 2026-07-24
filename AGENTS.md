# AGENTS.md — 项目指引

## 项目概述

本项目是「动手学深度学习」课程的学习与实践仓库，使用 PyTorch 从零实现经典模型并对比高级 API 版本。

- **语言**: Python 3.9
- **框架**: PyTorch（torch, torch.nn, torch.optim）, torchtext
- **辅助**: tqdm（进度条）, pandas（数据读取）, numpy
- **环境**: `.venv` 虚拟环境，`pyproject.toml` 管理依赖

## 项目结构

```
├── main.py                          # 入口（占位）
├── pyproject.toml                   # 项目配置
├── .python-version                  # Python 3.9
├── .venv/                           # 虚拟环境
├── DeepLearning_Project/
│   ├── BeginToCNN/                  # 1️⃣ 基础：回归 → MLP → CNN → LeNet
│   │   ├── 3_Regression.ipynb
│   │   ├── 4_MLP.ipynb
│   │   ├── 5_DeepLearningComputing.ipynb
│   │   ├── 6_Convolution.ipynb
│   │   └── 7_ModernLeNet.ipynb
│   ├── 8_RNN&9_ModernRNN/           # 2️⃣ RNN 及变体
│   │   ├── src/
│   │   │   ├── Config.py            # 超参数 + 模型选择
│   │   │   ├── Model.py             # 手写 RNN/GRU/LSTM + API 版本
│   │   │   ├── data_loader.py       # Shakespeare 数据集加载
│   │   │   ├── train.py             # 统一训练入口
│   │   │   └── generate.py          # 文本生成
│   │   └── data/
│   └── 10_Transformer/              # 3️⃣ Transformer 及现代架构
│       ├── Attention/
│       └── Encoder-Decoder/
│           ├── Config.py
│           ├── Seq2SeqModel.py      # Encoder + Decoder
│           ├── Data_loader.py       # 中英平行语料
│           └── train.py
└── MachineLearning_Project/         # 传统机器学习（待扩展）
    ├── RF/                          # 随机森林
    └── SVM/                         # 支持向量机
```

## 核心编码惯例

### 1. Config 类模式

每个模块有独立的 `Config` 类，所有超参数和路径通过类常量管理：

```python
class Config:
    BATCH_SIZE = 64
    EPOCH_NUM = 10
    LEARNING_RATE = 0.002
    EMBED_DIM = 256
    NUM_HIDDEN = 512
    MODEL_TYPE = "simple_gru"    # 模型选择字段
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
```

**规则**: 切换超参数只能改 Config.py，禁止在代码里硬编码数值。

### 2. 模型组织

每个模块的模型文件分两层：

- **手写实现**: 继承 `nn.Module`，用 `nn.Parameter` 显式定义权重矩阵。含中文注释解释公式原理。
- **API 版本**: 使用 `nn.RNN`/`nn.GRU`/`nn.LSTM` 等高级 API，加 `nn.Linear` 输出层。

六个模型统一接口：
```python
hidden = model.init_hidden(batch_size)       # → tensor 或 (H, C) 元组
outputs, hidden = model(inputs, hidden)      # 统一调用方式
```

### 3. 训练流程

标准训练循环模式：

```python
def train():
    device = Config.DEVICE
    train_loader, test_loader, vocab_size = get_data_loader(...)
    model = get_model(Config.MODEL_TYPE, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCH_NUM):
        model.train()
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            hidden = model.init_hidden(inputs.size(0))
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
```

### 4. 数据集处理

数据加载统一模式：

```python
class XxxDataset(Dataset):
    def __init__(self, sequences, targets): ...
    def __len__(self): ...
    def __getitem__(self, idx): ...

dataset = XxxDataset(...)
train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)
```

数据拆分使用 `torch.utils.data.random_split`。

### 5. 注释风格

- 使用**中文注释**解释 ML 概念和设计意图
- 关键公式和算法原理在注释中说明（如门控机制、注意力计算）
- 代码逻辑分块的注释使用 `# ----` 分隔

## 修改规则

1. **先读后改**: 修改前必须通读目标文件 + Config + Model + data_loader 三个关联文件
2. **最小变更**: 使用 `apply_patch` 做手术式编辑，保留原有逻辑、变量命名和结构
3. **Config 优先**: 任何可变的参数通过 Config 类管理，避免硬编码
4. **不要复制函数**: 切换模型/超参数通过 Config 字段，而非复制整段训练代码
5. **保持一致性**: 新增代码匹配现有的命名、缩进、注释风格

## 开发工作流

```
研究新模型/模块 → 创建主题文件夹 → Config.py → Model.py → data_loader.py → train.py
```

每个主题文件夹遵循 `{序号}_{名称}/` 命名模式，内部结构：
- `src/Config.py` — 超参数和路径
- `src/Model.py` — 模型定义（手写 + API）
- `src/data_loader.py` — 数据加载和预处理
- `src/train.py` — 训练入口
- `src/generate.py` — 推理/生成（可选）
- `data/` — 原始数据

## 可用资源

- **Skill**: `$deeplearning-skill` — 项目专属的详细修改指南和参考文档
- **环境**: `source .venv/bin/activate`
- **Jupyter notebooks**: `BeginToCNN/` 目录下的 `.ipynb` 文件用于实验和可视化
