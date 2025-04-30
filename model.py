import torch
import torch.nn as nn
import numpy as np

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # ✅ これが重要！

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x



class MiniGPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, dim_ff=256, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.decoder = TransformerDecoderLayer(d_model, nhead, dim_ff)
        self.out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.decoder(x, mask)
        return self.out(x)
