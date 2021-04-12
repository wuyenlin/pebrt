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

def align_sim3(model, data):
  """Implementation of the paper: S. Umeyama, Least-Squares Estimation
  of Transformation Parameters Between Two Point Patterns,
  IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
  Input:
  model -- first trajectory (3xn)
  data -- second trajectory (3xn)
  Output:
  s -- scale factor (scalar)
  R -- rotation matrix (3x3)
  t -- translation vector (3x1)
  t_error -- translational error per point (1xn)
  """

  # substract mean
  mu_M = model.mean(0).reshape(model.shape[0],1)
  mu_D = data.mean(0).reshape(data.shape[0],1)
  model_zerocentered = model - mu_M
  data_zerocentered = data - mu_D
  n = np.shape(model)[0]

  # correlation
  C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
  sigma2 = 1.0/n*np.multiply(data_zerocentered,data_zerocentered).sum()
  U_svd,D_svd,V_svd = np.linalg.linalg.svd(C)
  D_svd = np.diag(D_svd)
  V_svd = np.transpose(V_svd)
  S = np.eye(3)

  if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
    S[2,2] = -1

  R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))
  s = 1.0/sigma2*np.trace(np.dot(D_svd, S))
  t = mu_M-s*np.dot(R,mu_D)

  # TODO:
  # model_aligned = s * R * model + t
  # alignment_error = model_aligned - data
  # t_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

  return s, R, t #, t_error
