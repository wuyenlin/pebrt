import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.nn import init
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from common.hrnet import *

class PositionalEncoder(nn.Module):
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe
        if torch.cuda.is_available():
            self.pe = pe.cuda()
 
    def forward(self, x):
        bs, seq_len, d_model = x.size(0), x.size(1), x.size(2)
        x *= math.sqrt(d_model)
        pe = self.pe[:seq_len, :d_model]
        pe_all = pe.repeat(bs, 1, 1)

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return x

class TransformerEncoder(nn.Module):
    """
    Pose Estimation with Transformer
    """
    def __init__(self, d_model=34, nhead=2, num_layers=6, 
                    num_joints_in=17, num_joints_out=17):
        super().__init__()
        self.pe = PositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lin_out = nn.Linear(num_joints_in*2, num_joints_out*3, bias=False)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out 
        
    def forward(self, x):
        x = x.flatten(1).unsqueeze(1) #(bs,1,34)
        bs = x.size(0)
        x = self.pe(x)
        x = self.transformer(x)
        x = self.lin_out(x).squeeze(1)

        return x.reshape(bs, self.num_joints_out, 3)


class PETR_L(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HRNet(32, 17, 0.1)
        self.backbone.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))
        self.transformer = TransformerEncoder()
                                    
    def forward(self, x):
        x = self.backbone(x)
        x = hmap_joints(x)
        out_x = self.transformer(x.cuda())

        return out_x


if __name__ == "__main__":
    pass
    # model = HRNet(32, 17, 0.1)
    # model.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))