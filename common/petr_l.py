import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.nn import init
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from common.hrnet import HRNet

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.fc = nn.Sequential(
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(2048, 256),
            nn.Dropout(p=0.6, inplace=False),
            nn.Linear(256, 34),
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x

class PositionalEncoder(nn.Module):
    """
    Original PE from Attention is All You Need
    """
    def __init__(self, d_model, max_seq_len=10, dropout=0.1):
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

def normalize_screen_coordinates(X, w, h): 
    """
    referring to common/camera.py of facebookresearch/VideoPose3D 
    """
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

def get_joints(heatmap):
    """
    turn input heatmap (bs,17,64,48) into coordinates of 17 joints
    """
    assert heatmap.shape[1:] == (17,64,48), "{}".format(heatmap.shape)
    bs = heatmap.size(0)
    joints_2d = np.zeros([bs,17,2])
    heatmap = heatmap.cpu().detach().numpy()
    for i, human in enumerate(heatmap):
        for j, joint in enumerate(human):
            pt = np.unravel_index(np.argmax(joint), (64,48))
            joints_2d[i,j,:] = np.asarray(pt)
        joints_2d[i,:,:] = normalize_screen_coordinates(joints_2d[i,:,:], 48, 64)
    assert joints_2d.shape == (bs,17,2), "{}".format(joints_2d.shape)
    return torch.Tensor(joints_2d)

class PETR_L(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HRNet(32, 17, 0.1)
        self.backbone.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))
        self.transformer = TransformerEncoder()
                                    

    def forward(self, x):
        x = self.backbone(x)
        x = get_joints(x)
        out_x = self.transformer(x.cuda())

        return out_x


if __name__ == "__main__":
    pass
    # model = HRNet(32, 17, 0.1)
    # model.load_state_dict(torch.load('./weights/pose_hrnet_w32_256x192.pth'))