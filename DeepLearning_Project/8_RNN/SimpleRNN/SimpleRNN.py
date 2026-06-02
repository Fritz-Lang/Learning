import torch
from torch import nn

# Module 类是 torch.nn 模块里提供的一个模型构造类 (nn.Module)，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型
# PyTorch模型定义应包括两个主要部分：各个部分的初始化（__init__）；数据流向定义（forward）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, batch_size=64, device=device):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        
        # 不采用onehot编码，因为会把每个字符展成长度为len(vocab)的向量，导致维度爆炸
        # 词嵌入则可避免这一点，只会展成固定长度为embed_size的向量，远小于len(vocab)
        # 所以添加一个embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层
        self.W_xh = nn.Parameter(torch.rand(embed_dim, hidden_size, device = device) * 0.01)
        self.W_hh = nn.Parameter(torch.rand(hidden_size, hidden_size, device = device) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, device = device))

        # 全连接层
        self.W_hq = nn.Parameter(torch.rand(hidden_size, vocab_size, device = device) * 0.01)
        self.h_q = nn.Parameter(torch.zeros(vocab_size, device = device))

    def init_hidden(batch_size, hidden_size, device):
        return torch.zeros(batch_size, hidden_size, device)

    def forward(self, X):
        # 调用词嵌入层embedding，将输入展开成（批量大小，时间步数，嵌入维度）
        embedded = self.embedding(X)
        # 为了防止最后批量大小不足导致维度冲突，在每次循环计算前，取出批量大小
        batch_size, seq_len, _ = embedded.shape
        # 初始化隐状态
        H = self.init_hidden(batch_size, self.hidden_size, self.device)

        Outputs = []
        X_t = embedded @ self.W_xh
        # X按时间步取输入的切片
        for t in range(seq_len):
            H = torch.tanh(X_t[:, t, :] + torch.mm(H, self.W_hh) + self.b_h)
            O = torch.mm(H, self.W_hq) + self.h_q
            Outputs.append(O)

        return torch.cat(Outputs, dim=0), H