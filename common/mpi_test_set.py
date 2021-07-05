#!/usr/bin/python3
import scipy.io as sio
import numpy as np
import cv2 as cv
import os
import h5py
from tqdm import tqdm

def proc_mpi_test():
    """
    """
    files = ["./mpi_inf_3dhp_test_set/TS{}/annot_data.mat".format(S) for S in range(1,7)]
    img_paths = []
    positions_2d = []
    positions_3d = []
    print("Processing...")
    for i, char in enumerate(files):
        f = h5py.File(char, "r")
        img_paths[i] = {}
        positions_2d[i] = {}
        positions_3d[i] = {}
        kpts_3d = np.array(f["annot3"][:]).squeeze(1)
        kpts_2d = np.array(f["annot2"][:]).squeeze(1)
        valid = np.array(f["valid_frame"][:]).flatten()
        paths = ["./mpi_inf_3dhp_test_set/TS{}/imageSequence/img_{:06}.jpg".format(f) \
                for f in range(1, len(valid)+1)]

        for item in range(len(valid)):
            if valid[item] == 1:
                img_paths.append(paths[item])
                positions_2d.append(kpts_2d[item])
                positions_3d.append(kpts_3d[item])

    filename = "./data/mpi_inf_3dhp_test_set"
    np.savez_compressed(filename, img_paths=img_paths, \
            positions_2d=positions_2d, positions_3d=positions_3d)
    print("saved {}.npz".format(filename))
    print("Done.")

if __name__ == "__main__": 
    proc_mpi_test()