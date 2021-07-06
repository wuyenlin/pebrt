import cmath
import torch
from common.human import *


def pck(predicted, target):
    """
    Modified from sunnychencool/Anatomy3D
    (https://github.com/sunnychencool/Anatomy3D/blob/master/common/loss.py)
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    # threshold
    t = torch.Tensor([0.15]).cuda() if torch.cuda.is_available() else torch.Tensor([0.15])
    out = (dis < t).float() * 1
    return out.sum()/14.0


def auc(predicted, target):
    """
    Modified from sunnychencool/Anatomy3D
    (https://github.com/sunnychencool/Anatomy3D/blob/master/common/loss.py)
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    outall = 0
    for i in range(150):
        # threshold
        t = torch.Tensor([float(i)/1000]).cuda() if torch.cuda.is_available() \
            else torch.Tensor([float(i)/1000])
        out = (dis < t).float() * 1
        outall+=out.sum()/14.0
    outall = outall/150
    return outall


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    Borrowed from 'facebookresearch/VideoPose3D'.
    (https://github.com/facebookresearch/VideoPose3D/blob/master/common/loss.py)
    """
    assert predicted.shape == target.shape, "{}, {}".format(predicted.shape, target.shape)
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))



def is_so(M):
    det = cmath.isclose(torch.linalg.det(M), 1, rel_tol=1e-03)
    orth = cmath.isclose(torch.linalg.det(M@M.T), 1, rel_tol=1e-03)
    return 1 if orth and det else 2


def maev(predicted, target, w_kc=None):
    """
    MAEV: Mean Absolute Error of Vectors
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    :param w_kc: weight of kinematic constraints
    average error of 16 bones
    """
    bs, num_bones = predicted.size(0), predicted.size(1)
    predicted = predicted.view(bs,num_bones,3,3)
    target = target.view(bs,num_bones,3,3)
    w_orth = torch.ones(target.shape[:2])
    for b in range(bs):
        for bone in range(num_bones):
            M = predicted[b,bone]
            w_orth[b,bone] = is_so(M)
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w_orth = w_orth.cuda()
        w_kc = w_kc.cuda() if w_kc is not None else w_kc
    aev = torch.norm(torch.norm(predicted - target, dim=len(target.shape)-2), dim=len(target.shape)-2)
    maev = torch.mean(aev*w_kc) if w_kc is not None else torch.mean(aev)
    return maev


def mpbve(predicted, target, w_kc):
    """
    MPBVE - Mean Per Bone Vector Error
    Novel pose accuracy evaluation metric-
    Normalize each bone to 1m and calculate the mean L2 norms
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    """
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    bs, num_bones = predicted.shape[0], predicted.shape[1]

    pred_info = torch.zeros(bs, num_bones, 3)
    tar_info = torch.zeros(bs, num_bones, 3)

    for b in range(bs):
        pred = Human(1.8, "cpu")
        pred_model = pred.update_pose(predicted[b])
        pred_info[b,:] = vectorize(pred_model, "h36m")[:,:3]
        tar = Human(1.8, "cpu")
        tar_model = tar.update_pose(target[b])
        tar_info[b,:] = vectorize(tar_model, "h36m")[:,:3]
    mpbve = torch.mean(torch.norm(pred_info - tar_info, dim=len(tar_info.shape)-1))
    return mpbve
