#!/usr/bin/python3

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


def get_frame(file_list, k):
    img_path = "./dataset/S1/Seq1/imageSequence/video_0/" + file_list[k]
    return Image.open(img_path)


def animate(format="mp4"):
    fig = plt.figure()

    # animate dataset image stream
    ax1 = fig.add_subplot(121)
    file_list = sorted(os.listdir("./dataset/S1/Seq1/imageSequence/video_0/"))
    im = ax1.imshow(get_frame(file_list, 0))

    # animate 3D pose
    data = np.load("./pose_seq.npz", allow_pickle=True)
    data = data["arr_0"].reshape(1,-1)[0][0]
    ax2 = fig.add_subplot(122, projection='3d')

    # Initialize scatters
    scatters = [ ax2.scatter(data[0][p,0:1], data[0][p,1:2], data[0][p,2:]) for p in range(data[0].shape[0]) ]

    # Initialize lines
    bones = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,14), (11,12), (12,13),
        (2,11), (14,15), (15,16), # legs
        )

    lines_3d = [[] for _ in range(len(bones))]
    for n, bone in enumerate(bones):
        xS = (data[0][bone[0],0],data[0][bone[1],0])
        yS = (data[0][bone[0],1],data[0][bone[1],1])
        zS = (data[0][bone[0],2],data[0][bone[1],2])
        lines_3d[n].append(ax2.plot(xS, yS, zS))


    def update(iter, data, bones):
        im.set_data(get_frame(file_list, iter))
        for i in range(data[0].shape[0]):
            scatters[i]._offsets3d = (data[iter][i,0:1], data[iter][i,1:2], data[iter][i,2:])

        for n, bone in enumerate(bones):
            lines_3d[n][0][0].set_xdata(np.array([data[iter][bone[0],0],data[iter][bone[1],0]]))
            lines_3d[n][0][0].set_ydata(np.array([data[iter][bone[0],1],data[iter][bone[1],1]]))
            lines_3d[n][0][0].set_3d_properties(np.array([data[iter][bone[0],2],data[iter][bone[1],2]]), zdir="z")


    # Number of iterations
    iterations = len(data)
    print("number of frames:", iterations)
    print("Processing...")

    # Setting the axes properties
    ax2.set_xlim3d([-1.0, 1.0])
    ax2.set_xticklabels([])

    ax2.set_ylim3d([-1.0, 1.0])
    ax2.set_yticklabels([])

    ax2.set_zlim3d([-1.0, 1.0])
    ax2.set_zticklabels([])

    ax2.set_title('Reconstruction')
    ax2.view_init(elev=-90, azim=-90)

    anim = FuncAnimation(fig, update, iterations, fargs=(data, bones),
                                        interval=100, blit=False, repeat=False)

    if format == "mp4":
        Writer = writers['ffmpeg']
        writer = Writer(fps=10, metadata={})
        anim.save("output.mp4", writer=writer)
    elif format == "gif":
        anim.save("output.gif", dpi=80, writer='imagemagick')
    else:
        print("Unsupported file format")

    plt.close()


if __name__ == "__main__":
    animate()