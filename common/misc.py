import numpy as np
import torch
import cv2 as cv
import math, cmath
from math import sin, cos


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


def rot(a, b, r):
    """
    General rotation matrix
    a : yaw
    b : pitch
    r : roll
    returns a rotation matrix R given yaw, pitch, and roll angles
    """
    row1 = np.array([cos(a)*cos(b), cos(a)*sin(b)*sin(r)-sin(a)*cos(r), cos(a)*sin(b)*cos(r)+sin(a)*sin(r)])
    row2 = np.array([sin(a)*cos(b), sin(a)*sin(b)*sin(r)+cos(a)*cos(r), sin(a)*sin(b)*cos(r)-cos(a)*sin(r)])
    row3 = np.array([-sin(b), cos(b)*sin(r), cos(b)*cos(r)])
    R = np.array([row1, row2, row3])
    assert cmath.isclose(np.linalg.det(R), 1)
    return R


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Such that b = R @ a

    (Credit to Peter from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    angles = cv.RQDecomp3x3(R)[0]
    return R, angles 


def imshow(img):
    img = img / 2 + 0.5   
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    a = np.array([3,0,0])
    b = np.array([0,2,0])
    R, ang = rotation_matrix_from_vectors(a,b)
    print(R)
    c = R@a
    print(np.linalg.norm(c))
    c_hat = c/np.linalg.norm(c)
    b_hat = b/np.linalg.norm(b)
    print(c_hat)
    print(b_hat)
    print(ang)