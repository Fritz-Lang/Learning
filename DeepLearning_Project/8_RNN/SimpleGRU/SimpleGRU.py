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

        # 重置门参数
        self.W_xr = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        # 候选隐状态参数
        self.W_xh = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        # 更新门
        self.W_xz = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size)*0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def forward(self, X, H):
        embedded = self.embedding(X)
        Outputs = []
        
        seq_len = embedded[1]
        for t in range(seq_len):
            R_t = torch.sigmoid(embedded[:,t,:] @ self.W_xr + H @ self.W_hr + self.b_r)
            H_tilde = torch.tanh(embedded[:,t,:] @ self.W_xh + (R_t * H) @ self.W_hh + self.b_h)
            Z_t = torch.sigmoid(embedded[:,t,:] @ self.W_xz + H @ self.W_hz + self.b_z)
            H = Z_t @ H + (1 - Z_t) @ H_tilde
            O = H @ self.W_hq + self.b_q
            Outputs.append(O)
            
        return torch.cat(Outputs, dim=0), H