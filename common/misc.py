import numpy as np
import torch
import cv2 as cv
import math, cmath
from math import sin, cos
from scipy.spatial.transform import Rotation as R

def coco_mpi(coco_joints):
    """
    convert predicted COCO joints to MPI-INF-3DHP joints order
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


if __name__ == "__main__":
    pass
