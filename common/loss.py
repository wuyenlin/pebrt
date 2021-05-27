import math, cmath
from math import sin, cos
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def punish(predicted, bone1, bone2, weights, thres=0.1):
    """
    Verifies that symmetric bones have the same length.
    If the predicted skeleton shows different bone lengths on a pair of bones,
    one of their respective joints will be weighted 2 (instead of 1) in loss function.
    # TODO: add BONE collision (see if 2 lines/volumes have intersection)

    :param bone1:
    :param bone2:
    :return: a weight matrix of shape (bs, 17)
    """

# the commented part is an easier explanation of this function, 
# but training becomes slower as bs increases

    # for bs in range(predicted.shape[0]):
    #     len_bone1 = torch.norm(predicted[bs, bone1[0], :]-predicted[bs, bone1[1], :])
    #     len_bone2 = torch.norm(predicted[bs, bone2[0], :]-predicted[bs, bone2[1], :])
    #     if not math.isclose(len_bone1, len_bone2, abs_tol=1e-02):
    #         weights[bs, bone1[0]] = 2
    #         weights[bs, bone2[0]] = 2

    len1 = torch.norm(predicted[:, bone1[0], :]-predicted[:, bone1[1], :], dim=1)
    len2 = torch.norm(predicted[:, bone2[0], :]-predicted[:, bone2[1], :], dim=1)
    diff = abs(len2-len1) > thres
    diff = diff.double() + 1
    weights[:,bone1[0]], weights[:,bone1[1]] = diff, diff
    weights[:,bone2[0]], weights[:,bone2[1]] = diff, diff

    return weights


def joint_collision(predicted, target, weight, thres=0.1):
    """
    verify whether predicted and target joints lie within a given threshold
    if True -> collision -> punish

    :return: a weight matrix of shape (bs, 17)
    """
    bs = predicted.shape[0]
    diff = torch.linalg.norm(predicted - target, dim=2) < thres
    diff = diff.double() + 1
    weight *= diff

    return weight


def anth_mpjpe(predicted, target):
    """
    Added own implementation of weighted MPJPE by ensuring:
    1. symmetric bones have the same length
    2. wrists and ankles are correctly detected
    3. joint collision/occlusion are punished
    """
    assert predicted.shape == target.shape, "{}, {}".format(predicted.shape, target.shape)
    bs = predicted.shape[0]
    w = torch.ones(bs, predicted.shape[1]) # (bs,17)

    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w = w.cuda()
    
    bones = (((5,6),(8,9)),
             ((6,7),(9,10)),
             ((11,12),(14,15)),
             ((12,13),(15,16)))

    # 1. symmetric bones have same length, else punish
    for bone in bones:
        w = punish(predicted, bone[0], bone[1], w)

    # 2. focus on correct prediction of wrists and ankles
    w[:,7], w[:,10], w[:,13], w[:,16] = 2, 2, 2, 2

    # 3. joint collision
    w = joint_collision(predicted, target, w)

    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


def is_so(M):
    det = cmath.isclose(torch.linalg.det(M), 1, rel_tol=1e-03)
    orth = cmath.isclose(torch.linalg.det(M@M.T), 1, rel_tol=1e-03)
    return 1 if orth and det else 2


def maev(predicted, target):
    """
    MAEV: Mean Absolute Error of Vectors
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    """
    bs, num_bones = predicted.shape[0], predicted.shape[1]
    predicted = predicted.view(bs,num_bones,3,3)
    target = target.view(bs,num_bones,3,3)
    w_arr = torch.ones(target.shape[:2])
    for b in range(bs):
        for bone in range(num_bones):
            M = predicted[b,bone,:,:]
            w_arr[b,bone] = is_so(M)
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w_arr = w_arr.cuda()
    aev = torch.norm(torch.norm(predicted - target, dim=len(target.shape)-2), dim=len(target.shape)-2)
    maev = torch.mean(aev*w_arr)
    return maev


if __name__ == "__main__":
    a = torch.tensor([0.707,-0.707,0,0.707,0.707,0,0,0,1])
    a = a.repeat(16).reshape(1,16,9).to(torch.float32)
    b = torch.eye(3).flatten()
    b = b.repeat(16).reshape(1,16,9).to(torch.float32)
    print(maev(a,b))