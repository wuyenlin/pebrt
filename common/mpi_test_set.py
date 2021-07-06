#!/usr/bin/python3
import numpy as np
import cv2 as cv
import h5py


def convert_mpi(test_joints):
    """
    convert predicted MPI-INF-3DHP test set joint order
    to normal MPI-INF-3DHP one
    """
    # spine
    mpi_joints = np.zeros_like(test_joints)
    mpi_joints[0,:] = test_joints[1,:]
    mpi_joints[1,:] = test_joints[15,:]
    mpi_joints[2,:] = test_joints[14,:]
    mpi_joints[3,:] = test_joints[16,:]
    mpi_joints[4,:] = test_joints[0,:]
    # arms
    mpi_joints[5,:] = test_joints[5,:]
    mpi_joints[6,:] = test_joints[6,:]
    mpi_joints[7,:] = test_joints[7,:]
    mpi_joints[8,:] = test_joints[2,:]
    mpi_joints[9,:] = test_joints[3,:]
    mpi_joints[10,:] = test_joints[4,:]
    # legs
    mpi_joints[11,:] = test_joints[11,:]
    mpi_joints[12,:] = test_joints[12,:]
    mpi_joints[13,:] = test_joints[13,:]
    mpi_joints[14,:] = test_joints[8,:]
    mpi_joints[15,:] = test_joints[9,:]
    mpi_joints[16,:] = test_joints[10,:]

    return mpi_joints



def crop_square(img_path):
    """
    Given an image path open the file with OpenCV and get image dimension.
    Verify that all joints are within the frame, then save (or not).
    """
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    height = img.shape[0]
    width = img.shape[1]

    # check all kpts are within the (h,w)

def proc_mpi_test():
    files = ["./mpi_inf_3dhp_test_set/TS{}/annot_data.mat".format(S) for S in range(1,7)]
    img_paths = {}
    positions_2d = {}
    positions_3d = {}
    print("Processing...")
    for i, char in enumerate(files):
        f = h5py.File(char, "r")
        sub = "TS" + str(i+1)
        img_paths[sub] = []
        positions_2d[sub] = []
        positions_3d[sub] = []
        kpts_2d = np.array(f["annot2"][:]).squeeze(1)
        kpts_3d = np.array(f["annot3"][:]).squeeze(1)
        valid = np.array(f["valid_frame"][:]).flatten()
        paths = ["./mpi_inf_3dhp_test_set/TS{}/imageSequence/ \
                img_{:06}.jpg".format(i+1, f) for f in range(1, len(valid)+1)]

        for item in range(len(valid)):
            if valid[item] == 1:
                img_paths[sub].append(paths[item])
                positions_2d[sub].append(kpts_2d[item])
                positions_3d[sub].append(kpts_3d[item])

    merge_data = [img_paths, positions_2d, positions_3d]
    filename = "./mpi_inf_3dhp_test_set"
    np.savez_compressed(filename, img_paths=merge_data[0], \
            positions_2d=merge_data[1], positions_3d=merge_data[2])
    print("saved {}.npz".format(filename))
    print("Done.")


def try_read():
    npz_path = "./mpi_inf_3dhp_test_set.npz"
    data = np.load(npz_path, allow_pickle=True)
    print(data.files)
    print(data["positions_3d"].flatten()[0].keys())
    print(data["positions_2d"].flatten()[0].keys())
    print(data["img_paths"].flatten()[0].keys())


if __name__ == "__main__": 
    # proc_mpi_test()
    try_read()