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


class PEBRT(nn.Module):
    """
    PEBRT - Pose Estimation via Bone Rotation using Transformer
    """
    def __init__(self, device, bs=1, num_layers=2):
        super().__init__()
        
        self.bs = bs
        self.device = device
        self.transformer = TransformerEncoder(num_layers=num_layers).to(device)
        print("INFO: Using {} layers of Transformer Encoder.".format(num_layers))

 
    def normalize(self, x: torch.tensor) -> torch.tensor:
        return x/torch.linalg.norm(x)


    def gram_schmidt(self, arr) -> torch.tensor:
        """
        Detail implementation of Gram-Schmidt orthogonalization
        :param arr: a tensor of shape (16,6)
        :return Rs: a stack of flattened rotation matrix, i.e. (16,9)
        """
        import torch.nn.functional as F
        a_1, a_2 = arr[:,:3], arr[:,3:]
        row_1 = F.normalize(a_1, dim=1)
        dot = torch.sum((row_1*a_2),dim=1).unsqueeze(1)
        row_2 = F.normalize(a_2 - dot*row_1, dim=1)
        row_3 = torch.cross(row_1, row_2)
        R = torch.cat((row_1, row_2, row_3), 1) # stack + transpose
        R = R.view(-1,3,3).transpose(1,2)
        assert cmath.isclose(torch.linalg.det(R), 1, rel_tol=1e-04), torch.linalg.det(R)
        return R.reshape(-1,9)


    def process(self, arr_all):
        """
        an implementation of 6D representation for 3D rotation using Gram-Schmidt process
        1) project 6D to SO(3) via Gram-Schmidt process
        2) impose the recovered SO(3) on kinematic model and punish according to kinematic constraints

        :param arr: a (96,) tensor, 6D representation of 16 bones
        :return R_stack: (bs,16,9)
        :return w_kc: (bs, 16)
        """
        arr_all = arr_all.to(torch.float32).view(-1,16,6)
        R_stack = torch.zeros(arr_all.size(0),16,9)
        w_kc = torch.ones(arr_all.size(0),16)

        for b in range(arr_all.size(0)):
            arr = arr_all[b,:]
            assert arr.size(1) == 6
            R = self.gram_schmidt(arr)
            R_stack[b,:] = R.to(self.device)
            # Impose NN outputs SO(3) on kinematic model and get punishing weights
            h = Human(1.8, "cpu")
            h.update_pose(R_stack[b,:,:].flatten())
            w_kc[b,:] = torch.tensor(h.punish_list)
        return R_stack, w_kc


    def forward(self, x):
        x = self.transformer(x.float())
        x, w_kc = self.process(x)

        return x, w_kc

