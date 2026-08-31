import os
import sys
import torch
import pickle
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from Config import Config

DATA_FILE = os.path.join(Config.PROJECT_ROOT, "data")
TXT_FILE = os.path.join(DATA_FILE, "cmn-eng.txt")
ENGLISH_VOCAB_FILE = os.path.join(DATA_FILE, "ENGLISH_VOCAB_FILE")
CHINESE_VOCAB_FILE = os.path.join(DATA_FILE, "CHINESE_VOCAB_FILE")


def check_data_exists():
    '''检查数据文件是否存在'''
    if not os.path.exists(TXT_FILE):
        print("未找到文件")
        sys.exit(1)


def read_translation_data(path):
    '''数据清洗，列表包含对应字符串'''
    src_data = []
    tgt_data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip('\n').split('\t')
            src_data.append(line[0])
            tgt_data.append(line[1])
    return src_data, tgt_data


def tokenize(data):
    '''将字符串拆分成对应单词级列表'''
    tokenize_data = []

    punct = set(',.!?。？！，、')
    for line in data:
        tokens = []
        i = 0
        n = len(line)

        while i < n:
            ch = line[i]

            # 1. 如果是英文字母，连续取完作为一个单词
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                word = ''
                while i < n and (('a' <= line[i] <= 'z') or ('A' <= line[i] <= 'Z')):
                    word += line[i]
                    i += 1
                tokens.append(word)
                continue  # 跳过末尾的 i += 1

            # 2. 如果是标点，单独作为一个词元
            elif ch in punct:
                tokens.append(ch)
                i += 1
                continue

            # 3. 其他字符（包括中文、数字等），按单个字符拆分
            else:
                # 跳过空格（如果不想保留空格的话）
                if not ch.isspace():
                    tokens.append(ch)
                i += 1

        tokenize_data.append(tokens)

    return tokenize_data


def create_vocab(data, num_steps, vocab_path):
    '''根据文本列表构建对应词表'''
    tokenized_data = tokenize(data)
    listed_data = [
        token for line in tokenized_data for token in line]  # 二维列表展平

    counter = Counter(listed_data)
    tokens_freq = sorted(counter.items(), key=lambda x: x[1])
    freq2_text = [word for word, freq in tokens_freq if freq > 2]

    vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + freq2_text
    vocab_size = len(vocab)

    token_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_token = {i: ch for i, ch in enumerate(vocab)}

    with open(vocab_path, "wb") as file:
        pickle.dump((token_to_ix, ix_to_token, vocab_size), file)
    print(f"词表存入{vocab_path}, 词汇量大小{vocab_size}")

    unk_idx = token_to_ix['<unk>']
    # 将二维列表原文转为二维整数序列
    text_as_int = [[token_to_ix.get(ch, unk_idx)
                    for ch in line]for line in tokenized_data]

    bos_idx = token_to_ix['<bos>']
    eos_idx = token_to_ix['<eos>']
    pad_idx = token_to_ix['<pad>']

    # 超长序列只保留前 num_steps - 1 个词元，留一个位置给 <bos> 或 <eos>
    begin_seqs = []
    end_seqs = []

    for line in text_as_int:
        line = line[:num_steps - 1]

        # <bos> 放在序列首部，<eos> 放在序列尾部
        begin_seq = [bos_idx] + line
        end_seq = line + [eos_idx]

        # 右侧补齐 <pad> 到统一长度 num_steps
        if len(begin_seq) < num_steps:
            begin_seq += [pad_idx] * (num_steps - len(begin_seq))
        if len(end_seq) < num_steps:
            end_seq += [pad_idx] * (num_steps - len(end_seq))

        begin_seqs.append(begin_seq)
        end_seqs.append(end_seq)

    return begin_seqs, end_seqs, vocab_size, pad_idx


def prepare_data(sequence_length=8):
    '''加载、预处理数据并创建DataLoader'''
    check_data_exists()
    print(f"从{TXT_FILE}读取数据")
    src_data, tgt_data = read_translation_data(TXT_FILE)

    src_input_seqs, _, src_vocab_size, _ = create_vocab(
        src_data, sequence_length, ENGLISH_VOCAB_FILE)
    tgt_input_seqs, tgt_target_seqs, tgt_vocab_size, tgt_pad_idx = create_vocab(
        tgt_data, sequence_length, CHINESE_VOCAB_FILE)

    class TranslateDataset(Dataset):
        def __init__(self, src_input_seqs, tgt_input_seqs, tgt_target_seqs):
            self.src_input_seqs = src_input_seqs
            self.tgt_input_seqs = tgt_input_seqs
            self.tgt_target_seqs = tgt_target_seqs

        def __len__(self):
            return len(self.src_input_seqs)

        def __getitem__(self, idx):
            return (torch.tensor(self.src_input_seqs[idx], dtype=torch.long),
                    torch.tensor(self.tgt_input_seqs[idx], dtype=torch.long),
                    torch.tensor(self.tgt_target_seqs[idx], dtype=torch.long))

    dataset = TranslateDataset(src_input_seqs, tgt_input_seqs, tgt_target_seqs)

    return dataset, src_vocab_size, tgt_vocab_size, tgt_pad_idx


def get_data_loader(batch_size=64, sequence_length=8):
    dataset, src_vocab_size, tgt_vocab_size, tgt_pad_idx = prepare_data(
        sequence_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, src_vocab_size, tgt_vocab_size, tgt_pad_idx
