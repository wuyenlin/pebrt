import cmath
import torch


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
