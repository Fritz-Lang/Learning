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


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + num_hiddens,
                          num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, encoder_output):
        return encoder_output[1]

    def forward(self, X, H):
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
