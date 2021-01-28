#!/usr/bin/python3

import math
import torch, torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import models
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Backbone(nn.Module):
    """
    Implementing pretrained ResNet50 from PyTorch library
    """
    def __init__(self):
        super(Backbone, self).__init__()
        model_ft = models.resnet101(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv3 = nn.Conv2d(64, 1024, kernel_size=9, stride=4, padding=3, bias=False)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=7, stride=4, padding=3, bias=False)
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.conv3(x)
        x = self.conv5(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.embed_size = embed_size
        self.heads = heads

        self.tokeys = nn.Linear(embed_size, embed_size * heads, bias=False)
        self.toqueries = nn.Linear(embed_size, embed_size * heads, bias=False)
        self.tovalues = nn.Linear(embed_size, embed_size * heads, bias=False)

        self.unifyheads = nn.Linear(heads * embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):

        b, t, e = keys.size()
        h = self.heads

        values  = self.tovalues(values).view(b, t, h, e)
        keys    = self.tokeys(keys).view(b, t, h, e)
        queries = self.toqueries(queries).view(b, t, h, e)

        dot = torch.einsum('bthe,bihe->bhti', [queries, keys]) / math.sqrt(e)
        attention = F.softmax(dot, dim=2)

        if mask is not None: # mask out the upper half of the dot matrix, excluding the diagonal
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        out = torch.einsum('bhtd,bdhe->bthe', [attention, values]).reshape(b, t, h*e)

        return self.unifyheads(out)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        attention = self.attention(x)

        step = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(step)
        out = self.dropout(self.norm2(forward + x))

        return out

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__(self, src_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
        self.embed_size = embed_size
        self.embedding = nn.Embedding()
        self.position_embedding = nn.Embedding()

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.encoder = TransformerEncoder(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        # 2nd block od decoder takes two inputs from encoder
        out = self.encoder(value, key, query, src_mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self, emb):
        super().__init__()

def plot_heatmap(output):
    """
    output: Tensor
    """
    output = torch.einsum("chw->hw", [output]) / output.shape[0]
    out_cpu = output.cpu().detach().numpy()
    plt.imshow(out_cpu.reshape(64,64))
    cb = plt.colorbar()
    cb.set_label("number")
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bb = Backbone()
    bb = bb.to(device)

    img = Image.open("sample/01.jpg")
    transform = transforms.Compose([
        # transforms.Resize([64]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    x = transform(img)
    x = x.to(device)
    fe = bb(x.unsqueeze(0))
    fe = fe.squeeze(0)
    print(fe.shape)

    plot_heatmap(fe)
    plt.show

    net = SelfAttention(64)
    net = net.to(device)
    out = net(fe, fe, fe)
    print(out.shape)

    plot_heatmap(out)