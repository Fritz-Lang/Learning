import os
import pickle

import torch
from Config import Config
from Model import GRU_API


def generate_text(start_str="shall i compare thee to a summer's day", gen_length=500, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.path.join(Config.PROJECT_ROOT, "data")
    VOCAB_FILE = os.path.join(DATA_DIR, "vocab.pkl")
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "GRU_API_MODEL.pth")

    with open(VOCAB_FILE, 'rb') as f:
        char_to_ix, ix_to_char, vocab_size = pickle.load(f)

    model = GRU_API(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        num_hidden=Config.NUM_HIDDEN
    ).to(device)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    # 将数据中的权重字典应用到模型上
    model.load_state_dict(checkpoint['model_state_dict'])
    # 训练意外中断了，恢复优化器的动量状态
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"成功加载模型权重: {MODEL_SAVE_PATH}")

    # 评估模式，但并不关闭梯度计算
    model.eval()

    # 将起始字符串转为数字序列
    input_seq = [char_to_ix[ch] for ch in start_str]
    # 转为张量，需要注意的是，pytorch中一维张量 (n,) 会被默认视为列向量
    # .unsqueeze(0)：在第0维插入一维，形状变为 (1, seq_len)。这个 1 代表 batch_size
    # 这里可能会问，会不会seq_len和Config中的不一样导致报错？其实不会，因为每次模型只读入一个字符，循环直至结束。
    # (1, seq_len) -> (1, seq_len, embed_dim)，每次只读入(1, embed_dim)
    input_tensor = torch.tensor(
        input_seq, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    generated_text = start_str
    with torch.no_grad():
        for _ in range(gen_length):
            output, hidden = model(input_tensor, hidden)
            # 输出output(batch_size, vocab_size)
            # .squeeze(0)：删去第0维元素，形状变回(vocab_size, )
            # .div(temperature)：放大这种类别相似信息
            #
            output = output.squeeze(0).div(temperature).exp()
            # 加权随机抽样，抽一个样本
            # multinomial 返回的是一个一维张量（长度为 1），[0] 取出这个整数标量
            top_i = torch.multinomial(output, 1)[0]
            # 映射回字符
            predicted_char = ix_to_char[top_i.item()]
            generated_text += " " + predicted_char
            # 把刚生成的字符作为下一轮循环的输入
            input_tensor = torch.tensor([[top_i]], dtype=torch.long).to(device)

    print("\n--- 生成的文本 ---")
    print(generated_text)
    print("------------------\n")
