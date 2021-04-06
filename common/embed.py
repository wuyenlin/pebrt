import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms


class PositionalEncoder(nn.Module):
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, device, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model, device=device)
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

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return x



"""
Direction 3D pose regression method uses the model referring to Vision Transformer
Its implementation in PyTorch is availble at https://github.com/asyml/vision-transformer-pytorch.
Part of this file is borrowed from their src/model.py.
"""

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.H, self.W = self.img_size[0]/self.patch_size[0] , self.img_size[1]/self.patch_size[1] 
        self.num_patches = self.H * self.W
        self.in_channel = in_channel
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        H, W = h/self.patch_size[0] , w/self.patch_size[1]

        return x, (H,W)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding used in Vision Transformer (An Image is Worth 16x16 Words)
    """
    def __init__(self, num_patches, d_model, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, d_model))
        self.dropout = nn.Dropout(dropout) if dropout>0 else None

    def forward(self, x):
        x += self.pos_embed
        if self.dropout:
            x = self.dropout(x)
        
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pe = PositionalEmbedding(196, 768)
    a = torch.zeros(1,196,768)
    output = pe(a)
    plt.imshow(output[0].detach().numpy())
    plt.show()
