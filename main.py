import sys
import os

# 将子目录加入模块搜索路径
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "DeepLearning_Project", "8_RNN&9_ModernRNN", "src"))

from Config import Config
from train import train
from generate import generate_text


def run_training(model_type: str = "simple_gru"):
    """训练指定模型"""
    Config.MODEL_TYPE = model_type
    print(f"当前模型: {Config.MODEL_TYPE}")
    train()


def run_generation(model_type: str = "gru_api", prompt: str = "shall i compare thee to a summer's day?\n"):
    """用指定模型生成文本"""
    Config.MODEL_TYPE = model_type
    print(f"当前模型: {Config.MODEL_TYPE}")
    generate_text(start_str=prompt, gen_length=400, temperature=0.8)


if __name__ == "__main__":
    # 切换 Config.MODEL_TYPE 即可换模型运行
    # 可选: simple_rnn | simple_gru | simple_lstm | rnn_api | gru_api | lstm_api
    run_training("simple_gru")

    # 训练完后可取消下行注释来生成文本
    # run_generation("gru_api")
