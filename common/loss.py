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


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Borrowed from 'facebookresearch/VideoPose3D'.
    (https://github.com/facebookresearch/VideoPose3D/blob/master/common/loss.py)
    """
    assert predicted.shape == target.shape
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    


def punish(predicted, bone1, bone2, weights, thres=0.1):
    """
    This function verifies that symmetric bones have the same length.
    If the predicted skeleton shows different bone lengths on a pair of bones,
    one of their respective joints will be weighted 2 (instead of 1) in loss function.
    # TODO: add bone collision (see if 2 lines/volumes have intersection)

    ::return:: a weight matrix of shape (bs, 17)
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
    bs = predicted.shape[0]

    diff = torch.linalg.norm(predicted - target, dim=2) > thres
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

    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w = w.cuda()

    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


if __name__ == "__main__":
    a = torch.rand([4,17,3])
    b = torch.rand([4,17,3])
    print(anth_mpjpe(a,b))