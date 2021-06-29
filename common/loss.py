import cmath
import torch
from common.human import *


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


def joint_collision(predicted, target, weight, thres=0.1):
    """
    verify whether predicted and target joints lie within a given threshold
    if True -> collision -> punish

    :return: a weight matrix of shape (bs, 17)
    """
    diff = torch.linalg.norm(predicted - target, dim=2) < thres
    diff = diff.double() + 1
    weight *= diff

    return weight


def is_so(M):
    det = cmath.isclose(torch.linalg.det(M), 1, rel_tol=1e-03)
    orth = cmath.isclose(torch.linalg.det(M@M.T), 1, rel_tol=1e-03)
    return 1 if orth and det else 2


def maev(predicted, target, w_kc):
    """
    MAEV: Mean Absolute Error of Vectors
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    :param w_kc: weight of kinematic constraints
    average error of 16 bones
    """
    bs, num_bones = predicted.shape[0], predicted.shape[1]
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
        w_kc = w_kc.cuda()
        w_orth = w_orth.cuda()
    aev = torch.norm(torch.norm(predicted - target, dim=len(target.shape)-2), dim=len(target.shape)-2)
    maev = torch.mean(aev*w_kc)
    return maev


# 2. L2 norm on unit bone vectors

def mbve(predicted, target):
    """
    MBVE - Mean Bone Vector Error
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    """
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    bs, num_bones = predicted.shape[0], predicted.shape[1]

    pred_info = torch.zeros(bs, num_bones, 3)
    tar_info = torch.zeros(bs, num_bones, 3)

    pred = Human(1.8, "cpu")
    pred_model = pred.update_pose(predicted)
    tar = Human(1.8, "cpu")
    tar_model = tar.update_pose(target)
    for b in range(bs):
        pred_info[b,:] = vectorize(pred_model)[:,:3]
        tar_info[b,:] = vectorize(tar_model)[:,:3]
    mbve = torch.norm(pred_info - tar_info)
    return mbve


# 3. Decompose SO(3) into Euler angles

def meae(predicted, target):
    """
    MEAE: Mean Euler Angle Error
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    e.g. Decomposing a yields = (0,0,45) deg = (0,0,0.7854) rad
    sum of 3 ele is 0.7854, avg of 16 bones is 0.7854
    """
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    bs, num_bones = predicted.shape[0], predicted.shape[1]
    predicted = predicted.view(bs,num_bones,3,3)
    target = target.view(bs,num_bones,3,3)

    pred_euler = torch.zeros(bs,num_bones,3)
    tar_euler = torch.zeros(bs,num_bones,3)
    for b in range(bs):
        for bone in range(num_bones):
            pred_euler[b,bone,:] = abs(torch.tensor(rot_to_euler(predicted[b,bone])))
            tar_euler[b,bone,:] = abs(torch.tensor(rot_to_euler(target[b,bone])))
    return torch.mean(torch.sum(pred_euler - tar_euler, dim=2))
