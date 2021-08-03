import cmath
import torch


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
    len1 = torch.norm(predicted[:, bone1[0], :]-predicted[:, bone1[1], :], dim=1)
    len2 = torch.norm(predicted[:, bone2[0], :]-predicted[:, bone2[1], :], dim=1)
    diff = abs(len2-len1) > thres
    diff = diff.double() + 1
    weights[:,bone1[0]], weights[:,bone1[1]] = diff, diff
    weights[:,bone2[0]], weights[:,bone2[1]] = diff, diff

    return weights


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
