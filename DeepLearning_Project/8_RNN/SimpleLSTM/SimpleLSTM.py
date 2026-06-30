import torch
from torch import nn

class SimpleGRU(nn.Module):
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