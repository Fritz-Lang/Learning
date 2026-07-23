import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import re
import pickle

from Config import Config

# 数据位置
DATA_DIR = os.path.join(Config.PROJECT_ROOT, "data")
CSV_FILE = os.path.join(DATA_DIR, "Shakespeare_data.csv")
# 词表位置
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.pkl")

def check_data_exists():
    '''检查数据文件是否存在'''
    if not os.path.exists(CSV_FILE):
        print("未找到文件")
        sys.exit(1)

def read_shakespeare_data(path):
    '''读入台词数据集并加载到文本行的列表中'''
    #dataframe数据：print后会自动在最左侧添加一列作为行标识，但这个索引不是数据的一部分
    df = pd.read_csv(path)

    #Player列的值是NaN代表是舞台情景，需要剔除
    df_Player = df.dropna(subset=["Player"])

    #删除空台词，将每一行台词存入列表
    lines = df_Player["PlayerLine"].dropna().tolist()
    return [re.sub("[^A-Za-z]+", " ", line).strip() for line in lines]

def tokenize(lines, token="word"):
    '''将文本行拆分为单词或字符词元'''
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]

def prepare_data(sequence_length=50):
    '''加载、预处理数据并创建DataLoader'''
    #检查数据文件是否存在
    check_data_exists()

    #读入数据
    print(f"从{CSV_FILE}读取数据")
    lines = read_shakespeare_data(CSV_FILE)

    #单词词元
    tokens = tokenize(lines)
    #台词原文列表
    text = [token for line in tokens for token in line]

    #词表
    vocab = sorted(set(text))
    #词表大小
    vocab_size = len(vocab)

    #创建两个字典，一将单词转为数字，二将数字转回单词
    token_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_token = {i: ch for i, ch in enumerate(vocab)}

    #保存词汇表
    with open(VOCAB_FILE, "wb") as file:
        pickle.dump((token_to_ix, ix_to_token, vocab_size), file)
    print(f"词表存入{VOCAB_FILE}, 词汇量大小{vocab_size}")

    #台词原文转为整数序列
    text_as_int = [token_to_ix[ch] for ch in text]

    #输入序列和目标序列
    input_seqs = []
    target_seqs = []
    for i in range(0, len(text_as_int) - sequence_length):
        #输入序列长度=目标序列长度=sequence_length
        #共有（原文总长-训练序列长+1）个样本
        input_seqs.append(text_as_int[i: i + sequence_length])
        target_seqs.append(text_as_int[i + 1: i + 1 + sequence_length])

    class ShakespeareDataset(Dataset):
        '''
        所有的数据集想要在数据与标签之间建立映射,都需要继承Dataset类,
        所有的子类都需要重写__getitem__方法,
        该方法根据索引值获取每一个数据并且获取其对应的Label,
        子类也可以重写__len__方法,返回数据集的size大小
        '''
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets

        def __len__(self):
             return len(self.sequences)
        
        def __getitem__(self, idx):
             # 返回一个输入序列和对应的目标序列
             # 返回的形状为两个1*sequence_length的张量
             # 这个魔术方法可以使这个类能够像列表、字典一样用索引来设置元素
             return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx],dtype=torch.long)
        
    dataset = ShakespeareDataset(input_seqs, target_seqs)

    return dataset, vocab_size

def get_data_loader(batch_size = 64, sequence_length = 100):
    dataset, vocab_size = prepare_data(sequence_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # train_dataset和test_loader仍旧是ShakespeareDataset类，索引仍是返回两个值
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader本身不直接接收原始数据，它的输入必须是一个实现了__len__和__getitem__方法的Dataset对象
    # 对于ShakespeareDataset，每次next(迭代器)都会返回一个二元组
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_size