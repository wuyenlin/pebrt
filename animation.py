import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate():
    data = np.load("./pose_seq.npz", allow_pickle=True)
    data = data["arr_0"].reshape(1,-1)[0][0]
    pose = [data[frame].astype('float64') for frame in data.keys()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize scatters
    scatters = [ ax.scatter(data[0][p,0:1], data[0][p,1:2], data[0][p,2:]) for p in range(data[0].shape[0]) ]

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
        lines_3d[n].append(ax.plot(xS, yS, zS))


    def update(iteration, data, bones):
        for i in range(data[0].shape[0]):
            scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])

        for n, bone in enumerate(bones):
            lines_3d[n][0][0].set_xdata(np.array([data[iteration][bone[0],0],data[iteration][bone[1],0]]))
            lines_3d[n][0][0].set_ydata(np.array([data[iteration][bone[0],1],data[iteration][bone[1],1]]))
            lines_3d[n][0][0].set_3d_properties(np.array([data[iteration][bone[0],2],data[iteration][bone[1],2]]), zdir="z")


    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xticklabels([])

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_yticklabels([])

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zticklabels([])

    ax.set_title('Reconstruction')
    ax.view_init(elev=-80, azim=-90)

    ani = animation.FuncAnimation(fig, update, iterations, fargs=(data, bones),
                                        interval=100, blit=False, repeat=False)
    plt.show()
    plt.close()


if __name__ == "__main__":
    animate()