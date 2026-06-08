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
        # 词嵌入则可避免这一点，只会展成固定长度为embed_size的向量，远小于len(vocab)，所以添加一个embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层参数
        self.W_xh = nn.Parameter(torch.randn(embed_dim, hidden_size)*0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # 全连接层参数
        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size)*0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def init_hidden(self, batch_size, hidden_size, device=None):
        return torch.zeros(batch_size, hidden_size, device=device)

    def forward(self, X, H=None):
        # 这里的X不是一整个batch，只是batch的（输入序列，目标序列）的输入序列
        # 输入X形状（批量大小，时间步数）
        # 调用词嵌入层embedding，将输入展开成（时间步数，批量大小，嵌入维度）
        embedded = self.embedding(X.T)
        
        seq_len = embedded.shape[0]

        Outputs = []
        # X按时间步取输入的切片
        for t in range(seq_len):
            H = torch.tanh(torch.mm(embedded[t], self.W_xh) + torch.mm(H, self.W_hh) + self.b_h)
            O = torch.mm(H, self.W_hq) + self.b_q
            Outputs.append(O)

        # 输出O形状(时间步数，批量大小，词表大小)和传递到下一个的隐状态
        return torch.stack(Outputs, dim=0), H