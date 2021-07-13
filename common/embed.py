import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms


class PositionalEncoder(nn.Module):
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe
 
    def forward(self, x):
        bs, seq_len, d_model = x.size(0), x.size(1), x.size(2)
        x *= math.sqrt(d_model)
        pe = self.pe[:seq_len, :d_model]
        pe_all = pe.repeat(bs, 1, 1)
        pe_all = pe_all.to(x.device)

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        assert x.device == pe_all.device, "{},{}".format(x.device, pe_all.device)
        x += pe_all
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pe = PositionalEncoder(512, 1024)
    a = torch.zeros(1, 196, 512)
    output = pe(a)
    plt.imshow(output[0].detach().numpy())
    plt.show()
