import os
import numpy as np
from glob import glob
import cdflib
import sys
sys.path.append("../")

def merge_npz():
    merge_data = []
    files = ["./data_2d_h36m_gt.npz","./data_3d_h36m.npz"]
    print("Processing...")
    for item in files:
        t = np.load(item, allow_pickle=True)
        t = t["positions_2d"].reshape(1,-1) if item.endswith("2d_h36m_gt.npz") else t["positions_3d"].reshape(1,-1)
        merge_data.append(*t)
    filename = "./data_h36m"
    np.savez_compressed(filename, merge_data)
    print("saved {}.npz".format(filename))

output_filename = "data_3d_h36m"
output_filename_2d = "data_2d_h36m_gt"
subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
cdf_path = "./"

def cdf_to_npz(cdf_path, subjects, output_filename):
    print("Converting original Human3.6M dataset from", cdf_path, "(CDF files)")
    output = {}
    
    for subject in subjects:
        output[subject] = {}
        file_list = glob(cdf_path + "/" + subject + "/MyPoseFeatures/D3_Positions/*.cdf")
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == "S11" and action == "Directions":
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace("TakingPhoto", "Photo") \
                                    .replace("WalkingDog", "WalkDog")
            
            hf = cdflib.CDF(f)
            positions = hf["Pose"].reshape(-1, 32, 3)
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype("float32")
    
    print("Saving...")
    np.savez_compressed(output_filename, positions_3d=output)
    
    print("Done.")
        
    # Create 2D pose file
    # print("")
    # print("Computing ground-truth 2D poses...")
    # dataset = Human36mDataset(output_filename + ".npz")
    # output_2d_poses = {}
    # for subject in dataset.subjects():
    #     output_2d_poses[subject] = {}
    #     for action in dataset[subject].keys():
    #         anim = dataset[subject][action]
            
    #         positions_2d = []
    #         for cam in anim["cameras"]:
    #             pos_3d = world_to_camera(anim["positions"], R=cam["orientation"], t=cam["translation"])
    #             pos_2d = wrap(project_to_2d, pos_3d, cam["intrinsic"], unsqueeze=True)
    #             pos_2d_pixel_space = image_coordinates(pos_2d, w=cam["res_w"], h=cam["res_h"])
    #             positions_2d.append(pos_2d_pixel_space.astype("float32"))
    #         output_2d_poses[subject][action] = positions_2d
            
    # print("Saving...")
    # metadata = {
    #     "num_joints": dataset.skeleton().num_joints(),
    #     "keypoints_symmetry": [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    # }
    # np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    
    print("Done.")

if __name__ == "__main__":
    merge_npz()