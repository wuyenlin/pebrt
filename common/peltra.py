import torch
import torch.nn as nn
import cmath
from common.embed import *
from common.human import *



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
        self.transformer = TransformerEncoder(num_layers=8).to(device)

 
    def normalize(self, x: torch.tensor) -> torch.tensor:
        return x/torch.linalg.norm(x)


    def process(self, arr_all):
        """
        an implementation of 6D representation for 3D rotation using Gram-Schmidt process
        1) project 6D to SO(3) via Gram-Schmidt process
        2) impose the recovered SO(3) on kinematic model and punish according to kinematic constraints

        :param arr: a (96,) tensor, 6D representation of 16 bones
        :return R_stack: (bs,16,9)
        """
        arr_all = arr_all.to(torch.float32).view(-1,16,6)
        R_stack = torch.zeros(arr_all.shape[0],16,9)
        w_kc = torch.ones(arr_all.shape[0],16)
        for b in range(arr_all.shape[0]):
            for k in range(16):
                arr = arr_all[b,k,:]
                assert len(arr) == 6, len(arr)
                a_1, a_2 = arr[:3], arr[3:]
                row_1 = self.normalize(a_1)
                row_2 = self.normalize(a_2 - (row_1@a_2)*row_1)
                row_3 = self.normalize(torch.cross(row_1,row_2))
                R = torch.stack((row_1, row_2, row_3), 1) # SO(3)
                assert cmath.isclose(torch.linalg.det(R), 1, rel_tol=1e-04), torch.linalg.det(R)
                R_stack[b,k,:] = R.to(self.device).flatten()
            # Impose NN outputs SO(3) on kinematic model and get punishing weights
            h = Human(1.8, "cpu")
            h.update_pose(R_stack[b,:,:].flatten())
            w_kc[b,:] = torch.tensor(h.punish_list)
        return R_stack, w_kc


    def forward(self, x):
        x = self.transformer(x.float())
        x, w_kc = self.process(x)

        return x, w_kc
