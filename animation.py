# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def animate_scatters(iteration, data, scatters, lines, bones):

    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    for index in bones:
        xS = (data[0][index[0],0],data[0][index[1],0])
        yS = (data[0][index[0],1],data[0][index[1],1])
        zS = (data[0][index[0],2],data[0][index[1],2])
        lines.set_3d_properties([xS,yS,zS])
    return scatters

bones = (
    (2,1), (1,0), (0,3), (3,4),  # spine + head
    (0,5), (5,6), (6,7), 
    (0,8), (8,9), (9,10), # arms
    (2,14), (11,12), (12,13),
    (2,11), (14,15), (15,16), # legs
)


def viz(data, save=False):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

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
    for index in bones:
        xS = (data[0][index[0],0],data[0][index[1],0])
        yS = (data[0][index[0],1],data[0][index[1],1])
        zS = (data[0][index[0],2],data[0][index[1],2])
        lines = [ ax.plot(xS, yS, zS) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xticklabels([])
    # ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_yticklabels([])
    # ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zticklabels([])
    # ax.set_zlabel('Z')

    ax.set_title('Reconstruction')
    ax.view_init(elev=-80, azim=-90)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters, lines, bones),
                                       interval=100, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)

    plt.show()



if __name__ == "__main__":
    # data = generate_data(10, 2)
    # print(data[0].dtype)
    # print(len(data))
    # print(data[0].shape[0])
    # viz(data)
    data = np.load("./pose_seq.npz", allow_pickle=True)
    data = data["arr_0"].reshape(1,-1)[0][0]
    pose = [data[frame].astype('float64') for frame in data.keys()]
    # print(pose)
    # print(pose[0].dtype)
    # print(len(pose))
    # print(pose[0].shape[0])
    viz(pose)