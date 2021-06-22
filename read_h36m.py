import numpy as np
import h5py

def read_mat():
    filepath = "./Directions.54138969.mat"
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    print(arrays['Masks'][0])


def try_read():
    output = "./h36m/data_h36m_frame.npz"
    data = np.load(output, allow_pickle=True)
    print(data["arr_0"].reshape(1,-1)[0][0].keys())


def debug(plot=False):
    output_2d = "./h36m/data_2d_h36m_gt.npz"
    data_2d = np.load(output_2d, allow_pickle=True)
    
    photo = data_2d["positions_2d"].reshape(1,-1)[0][0]['S1']["Discussion"][0]
    print(data_2d["positions_2d"].reshape(1,-1)[0][0]['S1']['Photo'])
    first = photo[0,:,:]
    
    if plot: 
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        bones = (
            (0,7), (7,8), (8,9), (9,10),  # spine + head
            (8,14), (14,15), (15,16), 
            (8,11), (11,12), (12,13), # arms
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6) # legs
        )
        for p in first:
            ax.scatter(p[0], p[1], c='r', alpha=0.5)
        for index in bones:
            xS = (first[index[0]][0],first[index[1]][0])
            yS = (first[index[0]][1],first[index[1]][1])
            ax.plot(xS, yS)
        ax.set_xlim([0,1000])
        ax.set_xlabel("X")
        ax.set_ylim([0,1000])
        ax.set_ylabel("Y")
        ax.invert_yaxis()
        plt.show()


def count_frame():
    npz_path = "./data_3d_h36m.npz"
    data = np.load(npz_path, allow_pickle=True)
    data = data["positions_3d"].reshape(1,-1)[0][0]
    n_frame = 0
    for sub in data.keys():
        for action in data[sub].keys():
            n_frame += data[sub][action].shape[0]
    print(n_frame)


def viz():
    npz_path = "./data_3d_h36m.npz"
    data = np.load(npz_path, allow_pickle=True)
    photo = data["positions_3d"].reshape(1,-1)[0][0]['S1']["Photo"] 
    keep = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    final = np.zeros([1036,17,3])
    for f in range(1036):
        for row in range(17):
            final[f, row, :] = photo[f, keep[row], :]
        
    first = final[1000,:,:]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bones = (
        (0,7), (7,8), (8,9), (9,10),  # spine + head
        (8,14), (14,15), (15,16), 
        (8,11), (11,12), (12,13), # arms
        (0,1), (1,2), (2,3),
        (0,4), (4,5), (5,6) # legs
    )
    for p in first:
        ax.scatter(p[0], p[1], p[2], c='r', alpha=0.5)
    for index in bones:
        xS = (first[index[0]][0],first[index[1]][0])
        yS = (first[index[0]][1],first[index[1]][1])
        zS = (first[index[0]][2],first[index[1]][2])
        ax.plot(xS, yS, zS)
    ax.view_init(elev=0, azim=-90)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([0, 2.0])
    ax.set_zlabel("Z")
    plt.show()


def read():
    import cdflib
    output_2d = "./Poses_D2_Positions/Directions.54138969.cdf"
    output_2d = "./Directions.cdf"
    f = cdflib.CDF(output_2d)
    print(f.cdf_info())
    

if __name__ == "__main__":
    # try_read()
    # debug()
    read_mat()