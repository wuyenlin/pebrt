import os, json, csv, sys
from numpy.linalg import norm
import numpy as np 
from tqdm import tqdm


def merge_npz(full=False):
    merge_data = []
    for k in [0,1,2,4,5,6,7,8]:
        if full:
            npz = "dataset/S1/Seq1/imageSequence/full_video_{}.npz".format(k)
        else:
            npz = "dataset/S1/Seq1/imageSequence/video_{}.npz".format(k)
        t = np.load(npz, allow_pickle=True)
        t = t['arr_0'].reshape(1,-1)
        merge_data.append(*t)
    if full:
        np.savez_compressed("dataset/S1/Seq1/imageSequence/full_S1seq1", merge_data)
    else:
        np.savez_compressed("dataset/S1/Seq1/imageSequence/S1seq1", merge_data)
    print("saved")

if __name__ == "__main__":
    merge_npz(True)