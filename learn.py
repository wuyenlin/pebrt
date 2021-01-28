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

def plot_heatmap(output):
    """
    output: Tensor
    """
    output = torch.einsum("chw->hw", [output]) / output.shape[0]
    out_cpu = output.cpu().detach().numpy()
    plt.imshow(out_cpu.reshape(16, -1))
    cb = plt.colorbar()
    cb.set_label("number")
    plt.show()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        bb = Backbone()
        bb = bb.to(device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=10)
        model = transformer_encoder.to(device)

        img = Image.open("sample/01.jpg")
        transform = transforms.Compose([
            transforms.Resize([512]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        x = transform(img)
        plt.imshow(img)
        plt.show()
        x = x.to(device)
        fe = bb(x.unsqueeze(0))
        fe = fe.squeeze(0)
        print(fe.shape)
        plot_heatmap(fe)

        out = model(fe)
        print(out.shape)
        plot_heatmap(out)

