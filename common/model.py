import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.nn import init
from torchvision import transforms
from torchvision import models
from torchsummary import summary


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x

class PositionalEncoder(nn.Module):
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
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
        # x *= math.sqrt(d_model)
        pe = self.pe[:seq_len, :d_model]
        pe_all = pe.repeat(bs, 1, 1)

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return x

class TransformerEncoder(nn.Module):
    """
    Pose Estimation with Transformer
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, 
                    num_joints_in=17, num_joints_out=17):
        super().__init__()
        self.pe = PositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out 

        self.conv_layer = nn.Conv2d(d_model, d_model, 2, 2) 
        self.deconv_layer = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1) 
        self.final_layer = nn.Conv2d(d_model, num_joints_out*3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin_out = nn.Linear(256, num_joints_out*3)
        
    def forward(self, x):
        x = self.conv_layer(x)
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2,0,1) #(784, 8, 256)
        x = self.pe(x) 
        x = self.transformer(x) 
        x = x.permute(1, 2, 0).reshape(bs, c, h, w)
        x = self.avgpool(x)
        x = self.lin_out(x.squeeze(2).squeeze(2))

        # x = self.deconv_layer(x)
        # x = self.final_layer(x)

        return x.reshape(-1, self.num_joints_out, 3)


class PETR(nn.Module):
    def __init__(self, backbone, transformer):
        super().__init__()
        self.backbone = Backbone()
        self.transformer = TransformerEncoder()
                                    
    def forward(self, x):
        x = self.backbone(x)
        x = self.transformer(x)

        return x   


if __name__ == "__main__":

    model = PETR(Backbone, TransformerEncoder)
    model = model.cuda()
    summary(model, (3,224,224))