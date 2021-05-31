import torch
import torch.nn as nn
import cmath

try:
    from common.embed import *
except ModuleNotFoundError:
    from embed import *


class TransformerEncoder(nn.Module):
    """
    Pose Estimation with Transformer
    """
    def __init__(self, d_model=34, nhead=2, num_layers=8, 
                    num_joints_in=17, num_joints_out=17):
        super().__init__()

        print("INFO: Using default positional encoder")
        self.pe = PositionalEncoder(d_model)
        self.tanh = nn.Tanh()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lin_out = nn.Linear(d_model, 96)

        self.d_model = d_model
        self.nhead = nhead
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out 


    def forward(self, x):
        x = x.flatten(1).unsqueeze(1)
        x = self.pe(x)
        x = self.transformer(x)
        x = self.lin_out(x).squeeze(1)
        x = self.tanh(x)

        return x


class PELTRA(nn.Module):
    """
    PELTRA - Pose Estimation Lifting using TRansformer outputing Angles
    """
    def __init__(self, device, bs=1):
        super().__init__()
        
        self.bs = bs
        self.device = device
        self.transformer = TransformerEncoder(num_layers=8)

 
    def normalize(self, x: torch.tensor) -> torch.tensor:
        return x/torch.linalg.norm(x)


    def gs(self, arr_all: torch.tensor) -> torch.tensor:
        """
        an implementation of 6D representation for 3D rotation using Gram-Schmidt process
        project 6D to SO(3) via Gram-Schmidt process

        :param arr: a (96,) tensor, 6D representation of 16 bones
        :return R_stack: (bs,16,9)
        """
        R_stack = torch.zeros(self.bs,16,9)
        arr_all = arr_all.to(torch.float32).view(self.bs,16,-1)
        for b in range(self.bs):
            for k in range(16):
                arr = arr_all[b,k,:]
                assert len(arr) == 6, len(arr)
                a_1, a_2 = arr[:3], arr[3:]
                row_1 = self.normalize(a_1)
                row_2 = self.normalize(a_2 - (row_1@a_2)*row_1)
                row_3 = self.normalize(torch.cross(row_1,row_2))
                R = torch.stack((row_1, row_2, row_3), 1) # SO(3)
                assert cmath.isclose(torch.det(R), 1, rel_tol=1e-04), torch.det(R)
                R_stack[b,k,:] = R.to(self.device).flatten()
        return R_stack


    def forward(self, x):
        x = self.transformer(x.float())
        x = self.gs(x)

        return x


if __name__ == "__main__":
    a = torch.rand(2,17,2)
    model = PELTRA(device="cpu",bs=2)
    output = model(a)
    print(output.shape)
