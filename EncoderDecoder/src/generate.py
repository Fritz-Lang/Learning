import os
import pickle

import torch
from Config import Config
from Seq2SeqModel import EncoderDecoder, Seq2SeqDecoder, Seq2SeqEncoder


def generate_text(start_str="I love you", gen_length=500, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.path.join(Config.PROJECT_ROOT, "data")
    ENGLISH_VOCAB_FILE = os.path.join(DATA_DIR, "ENGLISH_VOCAB_FILE")
    CHINESE_VOCAB_FILE = os.path.join(DATA_DIR, "CHINESE_VOCAB_FILE")
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "EncoderDecoder_MODEL.pth")

    with open(ENGLISH_VOCAB_FILE, "rb") as f1:
        src_char_to_ix, _, src_vocab_size = pickle.load(f1)

    with open(CHINESE_VOCAB_FILE, "rb") as f2:
        tgt_char_to_ix, tgt_ix_to_char, tgt_vocab_size = pickle.load(f2)

    encoder = Seq2SeqEncoder(
        vocab_size=src_vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_hiddens=Config.NUM_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )
    decoder = Seq2SeqDecoder(
        vocab_size=tgt_vocab_size,
        embed_size=Config.EMBED_SIZE,
        num_hiddens=Config.NUM_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )
    model = EncoderDecoder(encoder, decoder).to(device)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    # 将数据中的权重字典应用到模型上
    model.load_state_dict(checkpoint["model_state_dict"])
    # 训练意外中断了，恢复优化器的动量状态
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"成功加载模型权重: {MODEL_SAVE_PATH}")

    # 评估模式，但并不关闭梯度计算
    model.eval()

    unk_idx = src_char_to_ix["<unk>"]
    bos_idx = tgt_char_to_ix["<bos>"]
    input_seq = [src_char_to_ix.get(ch, unk_idx) for ch in start_str.split()]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = ""
    with torch.no_grad():
        # 编码起始字符串，取编码器最终状态作为解码器初始状态
        encoder_output = model.encoder(input_tensor)
        dec_state = model.decoder.init_state(encoder_output)
        # 解码器从 <bos> 开始，逐词自回归生成
        dec_input = torch.tensor([[bos_idx]], dtype=torch.long).to(device)
        for _ in range(gen_length):
            output, dec_state = model.decoder(dec_input, dec_state)
            # 输出形状 (batch_size, seq_len, vocab_size)，单 token 输入取 (vocab_size, )
            output = output[0, 0].div(temperature).exp()
            # 加权随机抽样，抽一个样本
            # multinomial 返回的是一个一维张量（长度为 1），[0] 取出这个整数标量
            top_i = torch.multinomial(output, 1)[0]
            # 映射回字符
            predicted_char = tgt_ix_to_char[top_i.item()]
            # 生成到 <eos> 表示句子结束，提前终止
            if predicted_char == "<eos>":
                break
            generated_text += "" + predicted_char
            # 把刚生成的词作为下一轮解码器的输入
            dec_input = torch.tensor([[top_i]], dtype=torch.long).to(device)

    print("\n--- 生成的文本 ---")
    print(generated_text)
    print("------------------\n")


if __name__ == "__main__":
    generate_text()
