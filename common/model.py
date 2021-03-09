import os, math
import itertools
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=3, num_layers=3):
        """
        MLP or FFN mentioned in the paper
        (3-layer perceptron with 
            1. ReLU activation function
            2. hidden dimension d
            3. linear projection layer
        )
        In the end, FFN predicts the box coordinates (normalized), width and height;
                linear layer + Softmax predicts LABEL.

        Note: 
        input_dim = hidden_dim = transformer.d_model
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n,k)
                                    for n,k in zip([input_dim] + h, h + [output_dim]) )
        # same as creating 3 linear layers: 64,64; 64,64; 64,3
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
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
        x *= math.sqrt(d_model)
        pe = self.pe[:seq_len, :d_model]
        pe_all = pe.repeat(bs, 1, 1)

        assert x.shape == pe_all.shape, "{},{}".format(x.shape, pe_all.shape)
        x += pe_all
        return x

class tppe(nn.Module):
    """
    Implementation of 2D positional encoding used in TransPose
    """
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1, bs=128):
        super().__init__()
        self.bs = bs
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pe_x = Variable(torch.zeros(bs, max_seq_len, d_model), requires_grad=False).to(device)
        self.pe_y = Variable(torch.zeros(bs, max_seq_len, d_model), requires_grad=False).to(device)

        for i in range(0, bs, 2):
            for combi in itertools.zip_longest(range(d_model), range(max_seq_len)):
                if combi[0] is not None:
                    p_x = combi[0]
                    self.pe_x[i, :, p_x] = math.sin(2*math.pi*p_x/ (d_model* 10000 ** ((2 * i)/d_model)))
                    self.pe_x[i+1, :, p_x] = math.cos(2*math.pi*p_x/ (d_model* 10000 ** ((2 * (i+1))/d_model)))

                if combi[1] is not None:
                    p_y = combi[1]
                    self.pe_y[i, p_y, :] = math.sin(2*math.pi*p_y/ (max_seq_len * 10000 ** ((2 * i)/d_model)))
                    self.pe_y[i+1, p_y, :] = math.cos(2*math.pi*p_y/ (max_seq_len * 10000 ** ((2 * (i+1))/d_model)))
 

    def forward(self, x):

        bs, seq_len = x.size(0), x.size(1)
        pe_x = self.pe_x[:bs, :seq_len, :]
        pe_y = self.pe_y[:bs, :seq_len, :]

        assert x.shape == pe_x.shape, "{},{}".format(x.shape, pe_x.shape)
        assert x.shape == pe_y.shape, "{},{}".format(x.shape, pe_y.shape)
        x = x + pe_x + pe_y
        return x


class firstTransformer(nn.Module):
    def __init__(self, d_model=1, d_embed=64, nhead=1, num_layers=6, 
                    num_joints_in=15, num_joints_out=15,
                    dim_feedforward=2048, meth=2):
        super().__init__()
        if meth==1:
            self.pe = myPositionalEncoder(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.linear = nn.Linear(num_joints_in*2, num_joints_out*3, bias=False)
        elif meth==2:
            # self.lin_in = nn.Linear(d_model, d_embed, bias=False)
            self.pe = PositionalEncoder(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_embed, nhead, dim_feedforward)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.lin_out = nn.Linear(num_joints_out*2, num_joints_out*3, bias=False)

        self.meth = meth
        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, x, pe=True, meth=2):
        sz = x.shape[:3]
        if meth==1:
            x = torch.flatten(x, start_dim=2)
            if pe:
                x = self.pe(x) 
            x = self.transformer(x) 
            x = self.linear(x)
            # ran = [13, x.size(1)-13]
            # x = x[:, ran[0]:ran[1], :] # [b,n-26,45]

        elif meth==2:
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2,0,1) # (128,1,30) -> (30, 128, 1)
            if pe:
                x = self.pe(x) 
            x = self.transformer(x)
            x = x.permute(1,2,0) # (30, 128, 1) -> (128, 1, 30)
            x = self.lin_out(x)


        return x.reshape(sz[0], -1, self.num_joints_out, 3)

class LiftFormer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, 
                    num_joints_in=15, num_joints_out=15,
                    dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(num_joints_in*2, d_model)
        self.pe = PositionalEncoder(d_model)
        self.tpe = myPositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.liftformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, num_joints_out*3)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, x):
        sz = x.shape[:3]
        x = torch.flatten(x, start_dim=2) # from [b, n, 15, 2] to [b, n, 30]

        x = self.pe(self.linear_in(x))
        x = self.liftformer(x) 
        x = self.linear_out(self.dropout(x))


        return x.reshape(sz[0], -1, self.num_joints_out, 3)
        

class fullTransformer(nn.Module):
    def __init__(self, d_model=1, nhead=1,
                    num_enc_layers=6, num_dec_layers=6,
                    num_joints_in=15, num_joints_out=15,
                    dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pe = PositionalEncoder(d_model)

        self.embed = nn.Embedding(10, d_model)
        if torch.cuda.is_available():
            self.embed = nn.Embedding(10, d_model).cuda()

        self.transformer = nn.Transformer(d_model, nhead, 
                                    num_enc_layers, num_dec_layers,
                                    dim_feedforward, dropout)
        self.linear = nn.Linear(d_model, 1, bias=False)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, src):
        sz = src.shape[:3] #(128, 1, 15)
        
        src = self.pe(src.permute(2, 1, 0))
        tgt = torch.zeros(self.num_joints_out*3, sz[1], sz[0]).cuda()
        assert src.shape[1:] == tgt.shape[1:], "{},{}".format(src.shape, tgt.shape)

        out = self.transformer(src, tgt) # (45, rf, 128)
        out = out.permute(1, 2, 0)

        return out.reshape(sz[0], sz[1], self.num_joints_out, 3)