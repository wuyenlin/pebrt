
import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

try:
    from common.hrnet import *
    from common.pos_embed import *
except ModuleNotFoundError:
    from hrnet import *
    from pos_embed import *


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        # let d = 256 here
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        print(x.shape)
        x = self.conv(x) # (1, 2048, 16, 16) -> (1, 256, 16, 16)
        return x.flatten(2)  # (1, 256, 16*16) <- (bs, d, HW)


class fullTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4,
                    num_enc_layers=6, num_dec_layers=6,
                    num_joints_in=17, num_joints_out=17,
                    dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pe = PositionalEncoder(d_model)

        self.transformer = nn.Transformer(d_model, nhead, 
                                    num_enc_layers, num_dec_layers,
                                    dim_feedforward, dropout)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        
    
    def forward(self, src):
        #(bs,256,16*16)
        bs = src.size(0)
        out_feat = 100
        src = self.pe(src)
        src = src.permute(2,0,1) # (256,bs,256)
        tgt = torch.zeros(out_feat, bs, self.d_model).cuda()

        out = self.transformer(src, tgt) # (100, bs, 256)
        out = out.permute(1,0,2)

        return out.reshape(bs, out_feat, 16, 16)

class MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=51, num_layers=3):
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
        # same as creating 3 linear layers: 256,64; 64,64; 64,51
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = nn.Softmax(layer(x))

            else:
                x = layer(x)
        return x   


class myDETR(nn.Module):
    """
    my modified version of DETR
    """
    def __init__(self):
        super().__init__()
        
        self.backbone = Backbone()
        self.transformer = fullTransformer()
        self.mlp = MLP()
                                    

    def forward(self, x):
        x = self.backbone(x)
        x = self.transformer(x) 
        # src(256,bs,256)
        # tgt(100,bs,256)
        # out = self.mlp(x)

        return x


if __name__ == "__main__":
    transforms = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]) 
    # model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
    model = myDETR()
    model = model.cuda()
    img = Image.open("dataset/S1/Seq1/imageSequence/video_8/frame006192.jpg")
    img = transforms(img)
    img = img.unsqueeze(0)
    print(img.shape)
    img = img.cuda()
    output = model(img)
    print(output.shape)