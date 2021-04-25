import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from common.hrnet import *
    from common.embed import *
    from common.human import *
except ModuleNotFoundError:
    from hrnet import *
    from embed import *
    from human import *


class TransformerEncoder(nn.Module):
    """
    Pose Estimation with Transformer
    """
    def __init__(self, d_model=34, nhead=2, num_layers=6, 
                    num_joints_in=17, num_joints_out=17,
                    num_patches=256):
        super().__init__()

        print("INFO: Using default positional encoder")
        self.pe = PositionalEncoder(d_model)
        self.tanh = nn.Tanh()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lin_out = nn.Linear(d_model, 21)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out 


    def forward(self, x):
        x = x.flatten(1).unsqueeze(1) #(bs,1,34)
        x = self.pe(x)
        x = self.transformer(x)
        x = self.lin_out(x).squeeze(1) #(bs,21)
        x = self.tanh(x)
        x = torch.acos(x)

        return x


class PETRA(nn.Module):
    """
    PETRA - Pose Estimation using TRansformer outputing Angles
    """
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.backbone = HRNet(32, 17, 0.1)
        pretrained_weight = "../weights/pose_hrnet_w32_256x192.pth"
        self.backbone.load_state_dict(torch.load(pretrained_weight))
        print("INFO: Pre-trained weights of HRNet loaded from {}".format(pretrained_weight))
        self.transformer = TransformerEncoder(num_layers=8)


    def _decode_joints(self, heatmap):
        """
        turn input heatmap (bs,17,h,w) into coordinates of 17 joints
        return tensor of (17,2) joints on x,y coordinates for a batch
        """
        assert heatmap.shape[1:] == (17,64,64), "{}".format(heatmap.shape)
        bs = heatmap.size(0)
        joints_2d = np.zeros([bs,17,2])
        heatmap = heatmap.cpu().detach().numpy()

        for i, human in enumerate(heatmap):
            for j, joint in enumerate(human):
                pt = np.unravel_index(np.argmax(joint), (64,64))
                joints_2d[i,j,:] = np.asarray(pt, dtype=np.float64)
            joints_2d[i,:,:] = coco_mpi(joints_2d[i,:,:])
            # reposition root joint at origin
            joints_2d[i,:,:] = joints_2d[i,:,:] - joints_2d[i,2,:]
        # np.unravel_index gives (y,x) coordinates. need to swap it to (x,y)
        joints_2d[:,:,[0,1]] = joints_2d[:,:,[1,0]]
        return torch.tensor(joints_2d, device=self.device)


    def fk(self, x):
        """
        Perform Forward Kinematics on human skeleton model
        return (bs, 17, 3)
        """
        bs = x.size(0)
        prediction = np.zeros([bs,16,4])
        for i in range(bs):
            h = Human(1.7)
            model = h.update_pose(x[i, :])
            prediction[i,:,:] = vectorize(model)
        return torch.tensor(prediction, device=self.device, requires_grad=True)


    def forward(self, x):
        x = self.backbone(x)
        x = self._decode_joints(x)
        x = self.transformer(x.float())
        out_x = self.fk(x)

        return x, out_x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image

    transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]) 
    model = PETRA(device="cuda:0")
    model = model.cuda()
    img = Image.open("dataset/S1/Seq1/imageSequence/video_8/frame006192.jpg")
    img = transforms(img)
    img = img.unsqueeze(0)
    print(img.shape)
    img = img.cuda()
    _, output = model(img)
    print(output.shape)
