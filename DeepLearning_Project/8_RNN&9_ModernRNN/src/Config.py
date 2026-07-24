import os

class Config:
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 100
    EPOCH_NUM = 10
    LEARNING_RATE = 0.002
    EMBED_DIM = 256
    NUM_HIDDEN = 512

    # 可选: simple_rnn | simple_gru | simple_lstm | rnn_api | gru_api | lstm_api
    MODEL_TYPE = "simple_gru"

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
