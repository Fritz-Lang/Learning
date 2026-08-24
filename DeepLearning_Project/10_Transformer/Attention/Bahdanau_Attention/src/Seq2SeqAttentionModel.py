import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.gru(X)
        return output, state

# 无注意力的解码器，只是将编码器的最终隐状态作为上下文向量
# 然后将上下文向量和每一步输入拼接作为解码器的输入
# 但，实际上最终隐状态中有很多无需关注的信息
# 解码器的每一步输入只需要上下文向量中对应的信息即可
# 但上下文向量没法拆分
# 可以利用注意力机制，将源序列中我想要关注的某一步的信息重点注意
# 生成新的对应步的上下文向量
# 而且每一步上下文向量是由前一步的隐状态注意到的
# 所以，每一步的上下文向量是循环生成的
# 生成下一步的隐状态时，输入仍然是上下文向量与目标序列的输入拼接，然后和当前步的隐状态计算得出


def sequence_mask(X, valid_len, value=0.0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + num_hiddens,
                          num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, encoder_outputs, encoder_valid_lens):
        outputs, hidden_state = encoder_outputs
        return (outputs.permute(1, 0, 2), hidden_state, encoder_valid_lens)

    def forward(self, X, H):
        encoder_outputs, hidden_state, encoder_valid_lens = H
        X = self.embedding(X).permute(1, 0, 2)
        context = H[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.gru(X_and_context, H)
        output = self.dense(output).permute(1, 0, 2)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_X, decoder_X):
        encoder_output = self.encoder(encoder_X)
        dec_state = self.decoder.init_state(encoder_output)
        return self.decoder(decoder_X, dec_state)
