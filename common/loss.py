# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape, "{}, {}".format(predicted.shape, target.shape)
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def punish(predicted, bone1, bone2, weights):

# the commented code is an easier explanation of this function, 
# but training becomes slower as bs increases

    # for bs in range(predicted.shape[0]):
    #     len_bone1 = torch.norm(predicted[bs, bone1[0], :]-predicted[bs, bone2[1], :])
    #     len_bone2 = torch.norm(predicted[bs, bone2[0], :]-predicted[bs, bone2[1], :])
    #     if not math.isclose(len_bone1, len_bone2, abs_tol=1e-02):
    #         weights[bs, bone1[0]] = 2
    #         weights[bs, bone2[0]] = 2
    
    # return weights

    len1 = torch.norm(predicted[:, bone1[0], :]-predicted[:, bone1[1], :], dim=1)
    len2 = torch.norm(predicted[:, bone2[0], :]-predicted[:, bone2[1], :], dim=1)
    diff = abs(len2-len1) > 1e-02
    diff = list(map(int, diff))
    diff = torch.FloatTensor(diff) + 1
    weights[:,bone1[0]] = diff
    weights[:,bone2[0]] = diff

    return weights


def anth_mpjpe(predicted, target):
    assert predicted.shape == target.shape, "{}, {}".format(predicted.shape, target.shape)
    w = torch.ones(predicted.shape[0], predicted.shape[1]) # (64,17)
    
    bones = (((5,6),(8,9)),
             ((6,7),(9,10)),
             ((11,12),(14,15)),
             ((12,13),(15,16)))

    for bone in bones:
        w = punish(predicted, bone[0], bone[1], w)
    assert w.shape[0] == predicted.shape[0], "{},{}".format(w.shape, predicted.shape)

    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w = w.cuda()

    anth_mpjpe = torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))
    return anth_mpjpe


if __name__ == "__main__":
    a = torch.rand([16,17,2])
    b = torch.rand([16,17,2])
    print(mpjpe(a,b))