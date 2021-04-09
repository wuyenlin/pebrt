import os, math
import numpy as np
import torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from common.hrnet import *
    from common.embed import *
except ModuleNotFoundError:
    from hrnet import *
    from embed import *


class TransformerEncoder(nn.Module):
    """
    Pose Estimation with Transformer
    """
    def __init__(self, d_model=34, nhead=2, num_layers=6, 
                    num_joints_in=17, num_joints_out=17,
                    num_patches=256, lift=True):
        super().__init__()

        self.lift = lift
        if lift:
            print("INFO: Using default positional encoder")
            self.pe = PositionalEncoder(d_model)
            self.tanh = nn.Tanh()
        else:
            print("INFO: Using ViT positional embedding")
            self.pe = PositionalEmbedding(num_patches, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lin_out = nn.Linear(d_model, num_joints_out*3)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out 


    def forward(self, x):
        bs = x.size(0)
        if self.lift:
            x = x.flatten(1).unsqueeze(1) #(bs,1,34)
            x = self.pe(x)
            x = self.transformer(x)
            x = self.lin_out(x).squeeze(1)
            x = self.tanh(x)

        else:
            x = self.pe(x)   #(bs,196,768)
            x = self.transformer(x)
            x = self.lin_out(x[:,0])

        return x.reshape(bs, self.num_joints_out, 3)


class PETR(nn.Module):
    """
    PETR - Pose Estimation using TRansformer
    """
    def __init__(self, lift=True):
        super().__init__()
        
        self.lift = lift
        if self.lift:
            self.backbone = HRNet(32, 17, 0.1)
            pretrained_weight = "../weights/pose_hrnet_w32_256x192.pth"
            self.backbone.load_state_dict(torch.load(pretrained_weight))
            print("INFO: Pre-trained weights of HRNet loaded from {}".format(pretrained_weight))
            self.transformer = TransformerEncoder(num_layers=8)
        else:
            self.patch_embed = PatchEmbedding()
            self.transformer = TransformerEncoder(d_model=768, nhead=12, num_layers=12, lift=self.lift)
            self.joint_token = nn.Parameter(torch.zeros(1,1,768))
                                    

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
        assert joints_2d.shape == (bs,17,2), "{}".format(joints_2d.shape)
        # np.unravel_index gives (y,x) coordinates. need to swap it to (x,y)
        joints_2d[:,:,[0,1]] = joints_2d[:,:,[1,0]]
        return torch.tensor(joints_2d, device=heatmap.device)


    def forward(self, x):
        if self.lift:
            x = self.backbone(x)
            x = self._decode_joints(x)
            out_x = self.transformer(x.float())
        else:
            bs = x.shape[0]
            x = self.patch_embed(x)[0]
            joint_token = self.joint_token.repeat(bs,1,1)
            emb = torch.cat([joint_token, x], dim=1)
            out_x = self.transformer(emb)

        return x, out_x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image

    transforms = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]) 
    model = PETR(lift=True)
    model = model.cuda()
    img = Image.open("dataset/S1/Seq1/imageSequence/video_8/frame006192.jpg")
    img = transforms(img)
    img = img.unsqueeze(0)
    print(img.shape)
    img = img.cuda()
    _, output = model(img)
    print(output.shape)
