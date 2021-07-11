import numpy as np
from common.human import *
from common.dataloader import *
from common.misc import *

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms


transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
path = './h36m/data_h36m_frame_all.npz'
val_dataset = Data(path, transforms, False)
val_loader = DataLoader(val_dataset, batch_size=4, \
	shuffle=False, num_workers=16, drop_last=True, collate_fn=collate_fn)

dataiter = iter(val_loader)
img_path, inputs_2d, inputs3d, vec_3d = dataiter.next()
print(vec_3d[0].shape)
r = vec_3d[0].flatten()

h = Human(1.8, "cpu", "h36m")
model = h.update_pose(r)
print(h.punish_list)
vis_model(model)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

bones= (
	(2,1), (1,0), (0,3), (3,4),  # spine + head
	(0,5), (5,6), (6,7), 
	(0,8), (8,9), (9,10), # arms
        (2,11), (11,12), (12,13),
        (2,14), (14,15), (15,16), # legs
)

first = inputs3d[0]
for p in first:
	ax.scatter(p[0], p[1], p[2], c='r', alpha=0.5)
for index in bones:
	xS = (first[index[0]][0],first[index[1]][0])
	yS = (first[index[0]][1],first[index[1]][1])
	zS = (first[index[0]][2],first[index[1]][2])
	ax.plot(xS, yS, zS)
# ax.view_init(elev=90, azim=90)
ax.view_init(elev=20, azim=60)
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel("X")
ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel("Y")
ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel("Z")
plt.show()


def try_load():
    train_npz = "dataset/S1/Seq1/imageSequence/S1.npz"
    train_npz = "./h36m/data_h36m_frame_all.npz"
    train_dataset = Data(train_npz, transforms, True)
    trainloader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=False, num_workers=2, drop_last=True)
    print("data loaded!")
    dataiter = iter(trainloader)
    img_path, images, kpts, labels = dataiter.next()
    
    bones = (
        (2,1), (1,0), (0,3), (3,4),  # spine + head
        (0,5), (5,6), (6,7), 
        (0,8), (8,9), (9,10), # arms
        (2,11), (11,12), (12,13),
        (2,14), (14,15), (15,16), # legs
    )

    fig = plt.figure()

    # 2nd - 3D Pose
    pts = kpts[3]
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for bone in bones:
        xS = (pts[bone[0],0], pts[bone[1],0])
        yS = (pts[bone[0],1], pts[bone[1],1])
        zS = (pts[bone[0],2], pts[bone[1],2])
        
        ax.plot(xS, yS, zS, linewidth=3)
    ax.view_init(elev=20, azim=-95)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)

    # 3rd - vectorized 
    pts = vectorize(kpts[3])[:,:3]
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    for i in range(pts.shape[0]):
        xS = (0, pts[i,0])
        yS = (0, pts[i,1])
        zS = (0, pts[i,2])
        
        ax.plot(xS, yS, zS, linewidth=2)
    
    # unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='w', alpha=0.1)

    ax.view_init(elev=20, azim=-95)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)

    plt.show()

if __name__ == "__main__":

    from human import *
    from misc import *
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    try_load()