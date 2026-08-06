# AGENTS.md

## 项目基础信息

| 项目 | 说明 |
| --- | --- |
| 项目定位 | 李沐老师 D2L 课程的学习与实践仓库 |
| 内容范围 | 从线性回归到卷积神经网络为 D2L 代码复现；从循环神经网络开始为实践部分 |
| 开发语言 | Python 3.9 |
| 深度学习框架 | PyTorch |
| 辅助库 | `tqdm`、`pandas`、`numpy` |
| 环境管理 | `uv` |

## 项目结构

当前工作区结构如下，省略 `.venv/`、`.git/` 等环境目录：

```
.
├── .gitignore
├── .python-version                  # Python 版本：3.9.6
├── AGENTS.md                        # 项目协作与代码修改规范
├── README.md                        # 项目说明与快速开始
├── main.py                          # 统一入口
├── pyproject.toml                   # uv / PyTorch 项目配置
└── DeepLearning_Project/
    ├── BeginToCNN/                  # D2L 代码复现：线性回归 → CNN → LeNet
    │   ├── 3_Regression.ipynb
    │   ├── 4_MLP.ipynb
    │   ├── 5_DeepLearningComputing.ipynb
    │   ├── 6_Convolution.ipynb
    │   └── 7_ModernLeNet.ipynb
    ├── 8_RNN&9_ModernRNN/           # RNN / GRU / LSTM 实践
    │   ├── src/
    │   │   ├── Config.py
    │   │   ├── Model.py
    │   │   ├── data_loader.py
    │   │   ├── train.py
    │   │   └── generate.py
    │   └── data/
    │       ├── Shakespeare_data.csv
    │       └── vocab.pkl
    └── 10_Transformer/              # Transformer 实践
        ├── Attention/
        │   ├── Attention_Pooling/
        │   │   ├── config.py
        │   │   ├── model.py
        │   │   ├── train.py
        │   │   └── utils.py
        │   └── Scoring_Function/    # 空目录，待补充
        └── Encoder-Decoder/
            ├── data/
            │   └── cmn-eng.txt
            └── src/
                ├── Config.py
                ├── Seq2SeqModel.py
                ├── data_loader.py
                └── train.py
```

## 核心编码惯例

### 1. 编码前思考

- 明确假设，不确定时询问而非猜测。
- 存在歧义时，列出多种解释，不默默选定单一方案。
- 如果任务有明显更简单的做法，直接指出优化思路。
- 发现代码矛盾、逻辑不一致时及时暂停，请求信息澄清。

### 2. 简洁优先

- 用最少的代码解决问题，拒绝冗余实现。
- 不为一次性需求创建抽象层、复杂架构。
- 不盲目增加扩展性、可配置性，应对“未来可能用到”的场景。
- 若代码可大幅精简，主动重写优化。
- 校验标准：以资深工程师视角判断，代码若过于复杂，立即简化。

### 3. 精准修改

- 仅修改与当前任务直接相关的代码内容。
- 不顺手优化相邻代码、注释、排版格式。
- 不重构原本可以正常运行的代码模块。
- 严格匹配项目现有代码风格，保留原有编码习惯。
- 因本次修改产生的无效导入、废弃变量，可直接删除。
- 发现项目中原有的死代码、冗余内容，仅做文字提醒，不擅自删除。

### 4. 目标驱动执行

| 任务 | 成功标准 |
| --- | --- |
| 修复 Bug | 编写用例复现问题，再调试至用例正常通过 |
| 新增校验功能 | 针对异常输入编写测试用例，保证全部通过 |
| 代码重构 | 完成重构后，确保原有所有测试用例正常运行 |
| 多步骤复杂任务 | 先输出简短执行计划，同时标注每一步的验证方式 |

## 操作边界

- 新增依赖需先检查现有依赖，禁止重复引入同类库。
- 在修改代码之前可以展示哪些将被修改、修改后的样子，但正式修改必须得到同意才可修改。
