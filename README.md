# Learning

「动手学深度学习」课程实践仓库，基于 PyTorch 从零实现经典模型。

## 技术栈

Python 3.9.6 · PyTorch 2.8.0 · tqdm 4.67.3 · pandas 2.3.3 · numpy 2.0.2

环境管理使用 `uv`，依赖版本已精确锁定在 `pyproject.toml`。

## 项目结构

```
.
├── main.py                          # RNN 统一入口：训练 / 推理
├── pyproject.toml                   # uv / 依赖配置
├── BeginToCNN/                      # D2L 复现：线性回归 → MLP → 深度学习计算 → 卷积 → LeNet
│   ├── 3_Regression.ipynb
│   ├── 4_MLP.ipynb
│   ├── 5_DeepLearningComputing.ipynb
│   ├── 6_Convolution.ipynb
│   └── 7_ModernLeNet.ipynb
├── RNN/                             # RNN / GRU / LSTM 实践
│   ├── data/                        # 莎士比亚数据集与词表（不入库）
│   └── src/
│       ├── Config.py                # 超参数与模型选择
│       ├── Model.py                 # 手写 Simple* 与 API 版模型
│       ├── data_loader.py           # 数据加载与预处理
│       ├── train.py                 # 训练
│       └── generate.py              # 推理生成
└── Transformer/                     # Transformer 实践
    ├── Attention/
    │   ├── Attention_Pooling/       # 注意力汇聚（核回归等）
    │   └── Bahdanau_Attention/      # 注意力机制 + Seq2Seq
    └── Encoder-Decoder/             # 基础 Seq2Seq（中英翻译数据）
```

## 学习路线

| 阶段 | 模块 | 内容 |
|------|------|------|
| 1 | `BeginToCNN/` | 线性回归 → 多层感知机 → 深度学习计算 → 卷积 → 现代 LeNet |
| 2 | `RNN/` | RNN → GRU → LSTM，手写实现与 API 版对比 |
| 3 | `Transformer/` | 注意力机制 → Encoder-Decoder → Transformer |

各模块基本遵循 `Config + Model + data_loader + train` 的结构，通过 `Config.MODEL_TYPE` 切换模型。

## 快速开始

```bash
uv sync
source .venv/bin/activate

python main.py                 # 训练（模型保存至 RNN/data/）
python main.py --mode generate # 推理，生成莎士比亚风格文本
```

推理前需先完成一次训练（生成 `RNN/data/GRU_API_MODEL.pth`）。切换模型：修改 `RNN/src/Config.py` 的 `MODEL_TYPE`，可选 `rnn_api` / `gru_api` / `lstm_api`（`simple_*` 为学习手写版，暂不启用）。
