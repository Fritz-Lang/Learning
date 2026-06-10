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

    train_loader, _, vocab_size = get_data_loader(
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )

    model = SimpleRNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    
    # nn.CrossEntropyLoss(input, target)
    # input(N, C)，C为类别；target(N, )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练模型...")

    # 最外层循环：训练次数，在训练集上多次训练以优化模型
    for epoch in range(EPOCH_NUM):
        model.train()
        total_loss = 0
        hidden = None

        # 把progress理解为train_loader的一个带有进度展示的装饰器
        # 等价于for (inputs, targets) in train_loader
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCH_NUM}")
        # train_loader是一个迭代器
        # 每次循环都会获得下一个batch_size大小的数据（输入和目标序列）
        # 直到把训练集数据取完为止
        for (inputs, targets) in progress:
            # inputs&targets shape:(batch_size, seq_len)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            current_batch_size = inputs.size(0)
            hidden = model.init_hidden(current_batch_size, device)
            # output size:(seq_len, batch_size, vocab_size)
            outputs, hidden = model(inputs, hidden)

            # targets shape（batch_size，seq_len）展开为(batch_size*seq_len, )的一维张量
            targets = targets.view(-1)

            # 根据评价函数计算损失并反向传播+梯度裁剪+参数更新
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计损失，随时展示损失
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch + 1} 完成, 平均训练损失: {avg_loss:.4f}")