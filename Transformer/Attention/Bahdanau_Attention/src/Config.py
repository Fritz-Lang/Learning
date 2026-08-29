import os
import torch


class Config:
    EMBED_SIZE = 32
    NUM_HIDDEN = 64
    NUM_LAYERS = 2
    DROPOUT = 0.1
    BATCH_SIZE = 64
    NUM_STEPS = 8
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 10
    DEVICE = device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
