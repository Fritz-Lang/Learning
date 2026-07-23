import os
import sys
import torch
import collections
from torch.utils.data import Dataset, DataLoader
from Config import Config

TXT_FILE = os.path.join(Config.PROJECT_PATH, "cmn-eng.txt")
CLEANED_TXT_FILE = os.path.join(Config.PROJECT_PATH, "cleaned-cmn-eng.txt")

def check_data_exists():
    '''检查数据文件是否存在'''
    if not os.path.exists(TXT_FILE):
        print("未找到文件")
        sys.exit(1)

def data_clean():
    '''读入数据，返回英文序列和中文序列'''
    data_src = []
    data_tgt = []
    with open(TXT_FILE, "r", encoding='utf-8') as Fin:
        for line in Fin:
            line = line.strip()
            parts = line.split("\t")

            if len(parts) < 2:
                continue
            data_src.append(parts[0].strip())
            data_tgt.append(parts[1].strip())
    return data_src, data_tgt

def tokenize(data):
    '''将文本行拆分为单词词元'''
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

# 构建词表
def word_counter(tokenized_data):
    if (len(tokenized_data) == 0 or isinstance(tokenized_data[0], list)): 
        tokens = [token for line in tokenized_data for token in line]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self, tokenized_data, min_freq, special_tokens):
        counter = word_counter(tokenized_data)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = ['<unk>'] + special_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break  # 因为已经按频率降序排列，后面的词频只会更低，直接跳出
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小（包含特殊词元）"""
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        """根据输入获取索引：
           - 如果输入是单个词元，返回单个索引
           - 如果输入是列表或元组，返回索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            # 如果词不在词表中，返回 '<unk>' 对应的索引（即 0）
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """根据索引列表转换回词元列表：
           - 如果输入是单个索引，返回单个词元
           - 如果输入是列表或元组，返回词元列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引，固定为 0"""
        return 0

    @property
    def token_freqs(self):
        """返回已排序的词频列表（用于外部查看）"""
        return self._token_freqs

def prepare_data():
    '''加载、预处理数据并创建DataLoader'''
    check_data_exists()
    data_clean()

    print(f"从{CLEANED_TXT_FILE}读取数据")
    text_src, text_tgt = read_txt_data(CLEANED_TXT_FILE)

    #源序列和对应目标序列
    tokens_src = tokenize(text_src)
    tokens_tgt = tokenize(text_tgt)

    #词表
    eng_vocab, eng_vocab_size = create_vocab(tokens_src)
    cmn_vocab, cmn_vocab_size = create_vocab(tokens_tgt)

    #台词原文转为整数序列
    text_as_int = [token_to_ix[ch] for ch in text]

    class translateDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets

        def __len__(self):
             return len(self.sequences)
        
        def __getitem__(self, idx):
             return torch.tensor(self.sequences[idx], dtype=torch.long), \
                torch.tensor(self.targets[idx],dtype=torch.long)
        
    dataset = translateDataset(input_seqs, target_seqs)

    return dataset, vocab_size

def get_data_loader(batch_size = 64, sequence_length = 100):
    dataset, vocab_size = prepare_data(sequence_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_size