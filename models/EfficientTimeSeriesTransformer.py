import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]


class LinearAttention(nn.Module):
    def __init__(self, dim, nb_features=256):
        super().__init__()
        self.nb_features = nb_features
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


class EfficientTimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_linear_attn: bool = True):
        super().__init__()
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, input_dim)
        self.pos_enc = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.use_linear_attn = use_linear_attn
        if use_linear_attn:
            self.linear_attn = LinearAttention(input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        if self.use_linear_attn:
            x = self.linear_attn(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)
