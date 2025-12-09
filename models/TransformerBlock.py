import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, nhead=8, num_layers=2):
        super(TransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)

        return x
