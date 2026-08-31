import argparse
import os
import sys

# RNN 内部使用平铺导入（from Config import ...），需要把 RNN/src 加入模块搜索路径
RNN_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RNN", "src")
sys.path.insert(0, RNN_SRC)

from RNN.src.generate import generate_text
from RNN.src.train import train


def main():
    parser = argparse.ArgumentParser(description="RNN 训练 / 推理统一入口")
    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        default="train",
        help="train: 训练模型；generate: 推理生成文本",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        generate_text()


if __name__ == "__main__":
    main()
