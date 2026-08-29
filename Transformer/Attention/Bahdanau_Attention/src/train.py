import os
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from data_loader import get_data_loader
from Seq2SeqModel import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from Config import Config


def train():
    device = Config.DEVICE
    print(f"使用设备：{device}")

    train_loader, _, src_vocab_size, tgt_vocab_size, tgt_pad_idx = get_data_loader(
        batch_size=Config.BATCH_SIZE,
        sequence_length=Config.NUM_STEPS
    )

    encoder = Seq2SeqEncoder(
        vocab_size=src_vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_hiddens=Config.NUM_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    decoder = Seq2SeqDecoder(
        vocab_size=tgt_vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_hiddens=Config.NUM_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    model = EncoderDecoder(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print("开始训练模型...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        for (src, tgt_input, tgt_target) in progress:
            src, tgt_input, tgt_target = src.to(
                device), tgt_input.to(device), tgt_target.to(device)

            optimizer.zero_grad()
            outputs, _ = model(src, tgt_input)
            loss = criterion(outputs.permute(0, 2, 1), tgt_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计损失，随时展示损失
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch + 1} 完成, 平均训练损失: {avg_loss:.4f}")

        DATA_DIR = os.path.join(Config.PROJECT_ROOT, "data")
        MODEL_SAVE_PATH = os.path.join(DATA_DIR, "EncoderDecoder_MODEL.pth")

        torch.save(
            {
                'epoch': Config.NUM_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': Config
            },
            MODEL_SAVE_PATH
        )
        print(f"模型已保存至 {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train()
