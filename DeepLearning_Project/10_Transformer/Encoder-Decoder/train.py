import os
import re
import sys
import torch
from Config import Config

TXT_FILE = os.path.join(Config.PROJECT_PATH, "cmn-eng.txt")
CLEANED_TXT_FILE = os.path.join(Config.PROJECT_PATH, "cleaned-cmn-eng.txt")

def read_txt_data(path):
    '''按行读入数据，分解成对应的英汉句'''
    with open(path, "r", encoding="utf-8") as f:
        text_src = []
        text_tgt = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('\t')
            text_src.append(line[0])
            text_tgt.append(line[1])
        return text_src, text_tgt
    
text_src, text_tgt = read_txt_data(CLEANED_TXT_FILE)

def tokenize(text):
    '''将文本行拆分为单词词元'''
    tokenize_text = []

    punct = set(',.!?。？！，、')
    for line in text:
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
        
        tokenize_text.append(tokens)

    return tokenize_text

tokenize_text_src = tokenize(text_tgt)

print(f"{tokenize_text_src[:5]}")