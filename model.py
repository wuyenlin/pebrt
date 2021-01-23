import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import models
from torch.autograd import Variable
from PIL import Image

class my_net(nn.Module):
    """
    Implementing pretrained ResNet50 from PyTorch library
    """
    def __init__(self):
        super(my_net, self).__init__()
        model_ft = models.resnet101(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = model_ft


    def forward(self, x):
        x = self.model.conv1(x)
        # x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        return x



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    net = my_net()
    net = net.to(device)
    img = Image.open("01.jpg")
    x = TF.to_tensor(img)
    x.unsqueeze_(0)
    x = x.to(device)
    print("Input:")
    print(x.shape)

    out = net(x[0])
    print("Output:")
    print(out.shape)

    raw_weight = torch.bmm(out, out.transpose(1,2))
    print(raw_weight.shape)