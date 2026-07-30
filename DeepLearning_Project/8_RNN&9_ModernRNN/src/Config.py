import os

# 数据集：Kaggle上的莎士比亚戏剧数据集，包含超过11万行角色台词。
# 下载地址：https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays?resource=download

class Config:
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 100
    EPOCH_NUM = 10
    LEARNING_RATE = 0.002
    EMBED_DIM = 256
    NUM_HIDDEN = 512

    # 可选: simple_rnn | simple_gru | simple_lstm | rnn_api | gru_api | lstm_api
    # 暂不使用simple模型，由于模型为手写，参数不匹配
    MODEL_TYPE = "gru_api"

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
