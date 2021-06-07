from dataloader import *


def debug():
    train_npz = "dataset/S1/Seq1/imageSequence/S1Seq1.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=8, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
    
    bones = (
    (0,1), (0,3), (1,2), (3,4),  # spine + head
    (0,5), (0,8),
    (5,6), (6,7), (8,9), (9,10), # arms
    (2,14), (2,11),
    (11,12), (12,13), (14,15), (15,16), # legs
    )

    # 1st - Image
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(Image.open(img_path[0]))

    # 2nd- 3D Pose
    pts = kpts[0]
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        zS = (pts[bone[0],2], pts[bone[1],2])
        
        ax.plot(xS, yS, zS)
    
    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 3rd - change pose given rotation matrix to each bone
    h = Human(1.8, "cpu")
    pts = torch.tensor(labels[0])

    model = h.update_pose(pts, debug=True)
    model = model.detach().numpy()

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    for p in model:
        ax.scatter(p[0], p[1], p[2], c='r')

    for index in bones:
        xS = (model[index[0]][0], model[index[1]][0])
        yS = (model[index[0]][1], model[index[1]][1])
        zS = (model[index[0]][2], model[index[1]][2])
        ax.plot(xS, yS, zS)
    ax.view_init(elev=-80, azim=-90)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    debug()
