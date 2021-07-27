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
	shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)

dataiter = iter(val_loader)
img_path, inputs_2d, inputs3d, vec_3d = dataiter.next()
print(img_path[0])
r = vec_3d[0].flatten()

h = Human(1.8, "cpu", "h36m")
model = h.update_pose(r)
print(h.punish_list)
# vis_model(model)

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
	ax.scatter(p[0], p[1], p[2], c='r', linewidths=5)
for index in bones:
	xS = (first[index[0]][0],first[index[1]][0])
	yS = (first[index[0]][1],first[index[1]][1])
	zS = (first[index[0]][2],first[index[1]][2])
	ax.plot(xS, yS, zS, linewidth=5)
# ax.view_init(elev=90, azim=90)
ax.view_init(elev=10, azim=60)
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel("X")
ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel("Y")
ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel("Z")
# ax.grid(False)
plt.axis('off')
plt.show()
