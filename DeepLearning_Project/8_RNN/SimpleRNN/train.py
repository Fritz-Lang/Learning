import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from data_loader import get_data_loader
from SimpleRNN import SimpleRNN

# 超参数
BATCH_SIZE = 64
SEQUENCE_LENGTH = 100
EPOCH_NUM = 20
LEARNING_RATE = 0.0005
EMBED_DIM = 256
HIDDEN_SIZE = 512

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    train_loader, test_loader, vocab_size = get_data_loader(
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )

    model = SimpleRNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        batch_size=BATCH_SIZE,
        device=device
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练模型...")

    for epoch in range(EPOCH_NUM):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCH_NUM}")
        for i, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(BATCH_SIZE, HIDDEN_SIZE, device)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.zero_grad()

            output, hidden = model(inputs, hidden)

            # 将 targets 展平，以便与 output 的维度匹配
            targets = targets.view(-1)

            loss = loss_fn(output, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch + 1} 完成, 平均训练损失: {avg_loss:.4f}")

if __name__ == "__main__":
    train()