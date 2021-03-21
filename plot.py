import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from common.petr import PETR


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
]) 


def viz(bones):
    model = PETR(lift=True)
    model.load_state_dict(torch.load('./checkpoint/0321.bin'))
    model = model.cuda()
    img_list = [
            "dataset/S1/Seq1/imageSequence/video_4/frame001049.jpg",
            # "dataset/S1/Seq1/imageSequence/video_5/frame001182.jpg",
            ]
    k = 1
    fig = plt.figure()
    for i in range(len(img_list)):
        img_read = Image.open(img_list[i])
        img = transforms(img_read)
        img = img.unsqueeze(0)
        img = img.cuda()
        output = model(img)
        
        output = output.cpu().detach().numpy()
        ax = fig.add_subplot(2, len(img_list), k)
        ax.imshow(img_read)

        ax = fig.add_subplot(2, len(img_list), k+len(img_list), projection='3d')
        ax.scatter(output[:,:,0], output[:,:,1], output[:,:,2])
        for bone in bones:
            xS = (output[:,bone[0],0].tolist()[0], output[:,bone[1],0].tolist()[0])
            yS = (output[:,bone[0],1].tolist()[0], output[:,bone[1],1].tolist()[0])
            zS = (output[:,bone[0],2].tolist()[0], output[:,bone[1],2].tolist()[0])
            
            ax.plot(xS, yS, zS)
        ax.view_init(elev=-75, azim=-90)
        ax.autoscale()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        k += 1
    plt.show()


if __name__ == "__main__":
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )
    viz(bones)
