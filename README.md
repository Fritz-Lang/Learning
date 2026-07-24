# Learning

「动手学深度学习」课程实践仓库，基于 PyTorch 从零实现经典模型。

## 技术栈

Python 3.9 · PyTorch 2.x · tqdm · pandas · numpy

## 项目结构

```
├── main.py                          # 统一入口，一键训练
├── DeepLearning_Project/
│   ├── BeginToCNN/                  # 入门：线性回归 → MLP → CNN → LeNet
│   ├── 8_RNN&9_ModernRNN/           # RNN / GRU / LSTM（手写 + API 双版本）
│   └── 10_Transformer/              # Encoder-Decoder / Attention（进行中）
└── MachineLearning_Project/         # 传统机器学习（待扩展）
```

## 学习路线

按课程章节顺序推进：

| 阶段 | 模块 | 内容 |
|------|------|------|
| 1 | `BeginToCNN/` | 线性回归 → 多层感知机 → 深度学习计算 → 卷积 → 现代 LeNet |
| 2 | `8_RNN&9_ModernRNN/` | RNN → GRU → LSTM，含手写实现与 API 版对比 |
| 3 | `10_Transformer/` | 注意力机制 → Encoder-Decoder → Transformer |

每个模块统一遵循 Config + Model + data_loader + train 的结构，通过 `Config.MODEL_TYPE` 即可切换模型。

## 快速开始

```bash
source .venv/bin/activate
python main.py
```

切换模型：修改 `main.py` 中 `run_training()` 的模型名即可，可选 `simple_rnn` / `simple_gru` / `simple_lstm` / `rnn_api` / `gru_api` / `lstm_api`。
