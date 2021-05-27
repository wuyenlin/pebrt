#!/usr/bin/python3

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
# from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from common.petr import PETR
import cv2 as cv
import time
from tqdm import tqdm


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


def gen_kpts(vid_path):
    model = PETR(lift=True)
    model.load_state_dict(torch.load('./checkpoint/0321.bin'))
    model = model.cuda()
    

    cap = cv.VideoCapture(vid_path)
    total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # data = torch.zeros(total_frame, 17, 3).cuda()
    data = torch.zeros(50, 17, 3).cuda()
    k = 0
    while cap.isOpened():
        # start = time.time()
        ret, frame = cap.read()
        assert ret

        # cv.imshow("Frame", frame)
        frame = Image.fromarray(frame)
        frame = transforms(frame)
        frame = frame.unsqueeze(0).cuda()
        # print(out.shape)
        data[k, :, :] = model(frame)
        k += 1
        print("{}/{}".format(k,total_frame))
        if k == 50:
            break

        # fps = 1.0 / (time.time() - start)
        # print("FPS: {%.2f}" %(fps))
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break

    cap.release()
    cv.destroyAllWindows()
    return data.cpu().detach().numpy()

def update_pose(iteration, data, scatters):
    print("HI")
    for i in range(data.shape[0]):
        scatters[i,:,:]._offset3d = (data[iteration,:,0],data[iteration,:,1],data[iteration,:,2])
    return scatters


def main(data, bones, save=False):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(elev=-75., azim=-90)

    # Initialize scatters
    scatters = ax.scatter(data[0,:,0], data[0,:,1], data[0,:,2])
    for bone in bones:
        xS = (data[:,bone[0],0].tolist()[0], data[:,bone[1],0].tolist()[0])
        yS = (data[:,bone[0],1].tolist()[0], data[:,bone[1],1].tolist()[0])
        zS = (data[:,bone[0],2].tolist()[0], data[:,bone[1],2].tolist()[0])
        
        ax.plot(xS, yS, zS)

    # Number of iterations
    iterations = data.shape[0]

    # Setting the axes properties
    ax.set_xlim3d([-1.0,1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0,1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0,1.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Pose')

    ani = animation.FuncAnimation(fig, update_pose, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=False)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)

    plt.show()



if __name__ == "__main__":
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    vid_path = "walking.avi"
    data = gen_kpts(vid_path)
    main(data, bones)