import numpy as np
import torch


def coco_mpi(coco_joints):
    """
    convert predicted COCO joints to MPI-INF-3DHP joints order
    input is a (17,2) numpy array of COCO joints order
    returns a (17,2) numpy array of MPI-INF-3DHP order
    """
    mpi_joints = np.zeros_like(coco_joints)
    mpi_joints[4,:] = coco_joints[0,:]
    # arms
    mpi_joints[5,:] = coco_joints[6,:]
    mpi_joints[6,:] = coco_joints[8,:]
    mpi_joints[7,:] = coco_joints[10,:]
    mpi_joints[8,:] = coco_joints[5,:]
    mpi_joints[9,:] = coco_joints[7,:]
    mpi_joints[10,:] = coco_joints[9,:]
    # legs
    mpi_joints[11,:] = coco_joints[12,:]
    mpi_joints[12,:] = coco_joints[14,:]
    mpi_joints[13,:] = coco_joints[16,:]
    mpi_joints[14,:] = coco_joints[11,:]
    mpi_joints[15,:] = coco_joints[13,:]
    mpi_joints[16,:] = coco_joints[15,:]
    # spine (manual interpolation)
    mpi_joints[0,:] = (coco_joints[5,:] + coco_joints[6,:])/2
    mpi_joints[2,:] = (coco_joints[11,:] + coco_joints[12,:])/2
    mpi_joints[1,:] = (mpi_joints[0,:] + mpi_joints[2,:])/2
    mpi_joints[3,:] = (mpi_joints[0,:] + mpi_joints[4,:])/2

    return mpi_joints


def hmap_joints(heatmap):
    """
    turn input heatmap (bs,17,h,w) into coordinates of 17 joints
    return tensor of (17,2) joints on x,y coordinates for a batch
    """
    assert heatmap.shape[1:] == (17,64,64), "{}".format(heatmap.shape)
    bs = heatmap.size(0)
    joints_2d = np.zeros([bs,17,2])
    heatmap = heatmap.cpu().detach().numpy()
    for i, human in enumerate(heatmap):
        for j, joint in enumerate(human):
            pt = np.unravel_index(np.argmax(joint), (64,64))
            joints_2d[i,j,:] = np.asarray(pt)
        joints_2d[i,:,:] = coco_mpi(joints_2d[i,:,:])
        # reposition root joint at origin
        joints_2d[i,:,:] = joints_2d[i,:,:] - joints_2d[i,2,:]
    assert joints_2d.shape == (bs,17,2), "{}".format(joints_2d.shape)
    # np.unravel_index gives (y,x) coordinates. need to swap it to (x,y)
    joints_2d[:,:,[0,1]] = joints_2d[:,:,[1,0]]
    return torch.Tensor(joints_2d)

