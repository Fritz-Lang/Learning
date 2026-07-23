import torch
from torch import nn

# nn.Module类torch模块里提供的一个模型构造类 (nn.Module)，可以继承它来定义我们想要的模型
# 模型定义应包括两个主要部分：参数初始化（__init__方法）和前向传播（forward方法）

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        # 不采用onehot编码，因为会把每个字符展成长度为len(vocab)的向量，导致维度爆炸
        # 词嵌入则可避免这一点，只会展成固定长度为embed_size的向量，远小于len(vocab)
        # 所以添加一个embedding层，这个层将词汇表中的每个单词映射到一个embed_dim维的向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层参数
        self.W_xh = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # 全连接层将RNN的输出映射到词汇表的大小，从而预测下一个单词。
        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size)*0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def forward(self, X, H):
        # 输入X形状（批量大小，时间步数），调用词嵌入层embedding，将输入展开成（时间步数，批量大小，嵌入维度）
        embedded = self.embedding(X.T)

        Outputs = []
        # X按时间步取输入的切片
        seq_len = embedded.shape[0]
        for t in range(seq_len):
            H = torch.tanh(embedded[t] @ self.W_xh + H @ self.W_hh + self.b_h)
            O = H @ self.W_hq + self.b_q
            Outputs.append(O)

        # 输出O形状(时间步数，批量大小，词表大小)和传递到下一个的隐状态
        return torch.cat(Outputs, dim=0), H

# GRU
'''
# RNN对于每一次输入都是全部接收,不考虑这次输入有多少需要的
# GRU改进了这一点,用更新门来决定过去的隐状态和当前输入各需要保留多少信息
# 但当前输入不能直接和过去隐状态进行计算，要处理成一个隐状态来计算
# 那如何计算这个输入的隐状态呢?直接学习RNN隐状态处理当然可以,但是,这个隐状态中还有很多无用信息
# 所以，可以添加一个重置门，用当前输入来判断过去的隐状态有哪些信息要保留，处理成候选隐状态
# 最终，有了候选隐状态和过去隐状态，可以加权成新的隐状态了
# 当前输入的才是最重要的，用当前输入判断过去的隐状态哪些才是有用的，构建了候选隐状态，但不能否认候选隐状态可能出错，所以还要更新门
'''
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

# LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.W_xi = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.W_xf = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.W_xo = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.W_xc = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size)*0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))
 
    def init_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def forward(self, X, H, C):
        embedded = self.embedding(X)
        Outputs = []

        seq_len = embedded[1]
        for t in range(seq_len):
            I_t = torch.sigmoid(embedded[:,t,:] @ self.W_xi + H @ self.W_hi + self.b_i)
            F_t = torch.sigmoid(embedded[:,t,:] @ self.W_xf + H @ self.W_hf + self.b_f)
            O_t = torch.sigmoid(embedded[:,t,:] @ self.W_xo + H @ self.W_ho + self.b_o)
            C_tilde = torch.tanh(embedded[:,t,:] @ self.W_xc + H @ self.W_hc + self.b_c)
            C = F_t * C + I_t * C_tilde
            H = O_t * torch.tanh(C)
            O = H @ self.W_hq + self.b_q
            Outputs.append(O)

        return torch.cat(Outputs, dim=0), (H, C)

# 基于高级API搭建的RNN，GRU，LSTM
class RNN_API(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hidden = num_hidden

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, num_hidden, batch_first=True)
        self.fc = nn.Linear(num_hidden, vocab_size)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        # nn.RNN 默认 num_layers=1, bidirectional=False
        return torch.zeros(1, batch_size, self.num_hidden, device=device)

    def forward(self, X, H):
        X = self.embedding(X)
        out, H = self.rnn(X, H)
        out = out.contiguous().view(-1, self.num_hidden)
        out = self.fc(out)

        return out, H
    
class GRU_API(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hidden = num_hidden

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, num_hidden, batch_first=True)
        self.fc = nn.Linear(num_hidden, vocab_size)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        # nn.GRU 的输入 h_0 形状必须是 (num_layers * num_directions, batch, hidden_size)
        # 形状：(num_layers, batch, num_hidden)，这里 num_layers=1
        return torch.zeros(1, batch_size, self.num_hidden, device=device)

    def forward(self, X, H):
        X = self.embedding(X)
        out, H = self.gru(X, H)
        out = out.contiguous().view(-1, self.num_hidden)
        out = self.fc(out)

        return out, H

class LSTM_API(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hidden = num_hidden

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, num_hidden, batch_first=True)
        self.fc = nn.Linear(num_hidden, vocab_size)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        # LSTM 需要同时初始化 H（隐藏状态）和 C（细胞状态）
        # 形状都是 (num_layers, batch, num_hidden)，这里 num_layers=1
        h0 = torch.zeros(1, batch_size, self.num_hidden, device=device)
        c0 = torch.zeros(1, batch_size, self.num_hidden, device=device)
        return h0, c0  # 返回一个元组

    def forward(self, X, state):
        # state 是一个元组 (H, C)
        X = self.embedding(X)
        
        # LSTM 的输入：X 和 (H, C)；输出：out 和 (H_new, C_new)
        out, (h_new, c_new) = self.lstm(X, state)
        out = out.contiguous().view(-1, self.num_hidden)
        out = self.fc(out)

        return out, (h_new, c_new)