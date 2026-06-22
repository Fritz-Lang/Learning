# RNN对于每一次输入都是全部接收，不考虑这次输入有多少需要的
# GRU改进了这一点，用更新门来决定过去的隐状态和当前输入各需要保留多少信息
# 但当前输入不能直接和过去隐状态进行计算，要处理成一个隐状态来计算
# 那如何计算这个输入的隐状态呢？直接学习RNN隐状态处理当然可以，但是，这个隐状态中还有很多无用信息
# 所以，可以添加一个重置门，用当前输入来判断过去的隐状态有哪些信息要保留，处理成候选隐状态
# 最终，有了候选隐状态和过去隐状态，可以加权成新的隐状态了

# 当前输入的才是最重要的，用当前输入判断过去的隐状态哪些才是有用的，构建了候选隐状态，但不能否认候选隐状态可能出错，所以还要更新门

import torch
from torch import nn

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.