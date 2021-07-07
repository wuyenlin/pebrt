import cv2 as cv
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from common.dataloader import *
from common.petr import PETR
from common.human import *


def extract_bone(pred, bone, k):
    out = (pred[:,bone[0],k].tolist()[0], pred[:,bone[1],k].tolist()[0])
    return out


def plot3d(ax, output):
    bones = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,14), (11,12), (12,13),
        (2,11), (14,15), (15,16) # legs
    )
    ax.scatter(output[:,:,0], output[:,:,1], output[:,:,2])
    for bone in bones:
        xS = extract_bone(output, bone, 0)
        yS = extract_bone(output, bone, 1)
        zS = extract_bone(output, bone, 2)
        ax.plot(xS, yS, zS)
    ax.view_init(elev=5, azim=90)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel("Z")


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


def get_frame():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # img = Image.fromarray(frame)
        # cv.imshow("frame", transforms(img))
        # if cv.waitKey(1) == ord("q"):
        #     break

    cap.release()
    cv.destroyAllWindows()
    return Image.fromarray(frame)


def animate(model):
    # plot 3D output 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # using opencv to get frame + cv.imshow webcam
    frame = transforms(frame)
    output = model(frame)
    output = output.cpu().detach().numpy()


    # Initialize scatters
    scatters = [ ax.scatter(output[p,0:1], output[p,1:2], output[p,2:]) for p in range(data[0].shape[0]) ]

    # Initialize lines
    bones = (
        (0,7), (7,8), (8,9), (9,10),  # spine + head
        (8,14), (14,15), (15,16), 
        (8,11), (11,12), (12,13), # arms
        (0,1), (1,2), (2,3),
        (0,4), (4,5), (5,6) # legs
        )

    lines_3d = [[] for _ in range(len(bones))]
    for n, bone in enumerate(bones):
        xS = (output[bone[0],0], output[bone[1],0])
        yS = (output[bone[0],1], output[bone[1],1])
        zS = (output[bone[0],2], output[bone[1],2])
        lines_3d[n].append(ax.plot(xS, yS, zS))


    def update(iter, data, bones):
        # im.set_data(get_frame(file_list, iter))

        # using opencv to get frame + cv.imshow webcam
        frame = transforms(frame)
        output = net(frame)
        output = output.cpu().detach().numpy()

        for i in range(data[0].shape[0]):
            scatters[i]._offsets3d = (data[iter][i,0:1], data[iter][i,1:2], data[iter][i,2:])

        for n, bone in enumerate(bones):
            lines_3d[n][0][0].set_xdata(np.array([data[iter][bone[0],0],data[iter][bone[1],0]]))
            lines_3d[n][0][0].set_ydata(np.array([data[iter][bone[0],1],data[iter][bone[1],1]]))
            lines_3d[n][0][0].set_3d_properties(np.array([data[iter][bone[0],2],data[iter][bone[1],2]]), zdir="z")


    iterations = len(output)

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xticklabels([])

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_yticklabels([])

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zticklabels([])

    ax.set_title("Reconstruction")
    ax.view_init(elev=-90, azim=-90)

    anim = FuncAnimation(fig, update, iterations, fargs=(output, bones),
                                        interval=100, blit=False, repeat=False)

    # if format == "mp4":
    #     Writer = writers["ffmpeg"]
    #     writer = Writer(fps=10, metadata={})
    #     anim.save("output.mp4", writer=writer)

    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PETR(device)
    net.load_state_dict(torch.load("./peltra/ft_1_h36m.bin")["model"])
    net = net.to(device)
    net.eval()
    animate(model=net)