from PIL import Image
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
bones= (
	(2,1), (1,0), (0,3), (3,4),  # spine + head
	(0,5), (5,6), (6,7), 
	(0,8), (8,9), (9,10), # arms
	(2,11), (11,12), (12,13),
	(2,14), (14,15), (15,16), # legs
)

def plot_3d(ax, bones, output, color="red"):
	for p in output:
		if color == "blue":
			ax.scatter(p[0], p[1], p[2], c='b', alpha=0.4)
		else:
			ax.scatter(p[0], p[1], p[2], c='r', alpha=0.5)
	for index in bones:
		xS = (output[index[0]][0],output[index[1]][0])
		yS = (output[index[0]][1],output[index[1]][1])
		zS = (output[index[0]][2],output[index[1]][2])
		ax.plot(xS, yS, zS, linewidth=5)
	ax.view_init(elev=20, azim=60)
	ax.set_xlim3d([-1.0, 1.0])
	ax.set_xlabel("X")
	ax.set_ylim3d([-1.0, 1.0])
	ax.set_ylabel("Y")
	ax.set_zlim3d([-1.0, 1.0])
	ax.set_zlabel("Z")

path = './h36m/data_h36m_frame_all.npz'
val_dataset = Data(path, transforms, False)
val_loader = DataLoader(val_dataset, batch_size=4, \
	shuffle=True, num_workers=16, drop_last=True, collate_fn=collate_fn)

dataiter = iter(val_loader)
img_path, inputs_2d, inputs3d, vec_3d = dataiter.next()
r = vec_3d[0].flatten()

# h = Human(1.8, "cpu", "h36m")
# model = h.update_pose(r)
# print(h.punish_list)
# vis_model(model.detach().cpu().numpy())

fig = plt.figure()

ax = fig.add_subplot(221)
plt.imshow(Image.open(img_path[0]))
plt.title("Original image", fontsize=35)

ax = fig.add_subplot(222, projection='3d')
first = inputs3d[0]
plot_3d(ax, bones, first, "blue")
plt.title("GT 3D keypoints", fontsize=35)

ax = fig.add_subplot(223, projection='3d')
h = Human(1.9, "cpu", "h36m")
model_1 = h.update_pose(r)
plot_3d(ax, bones, model_1.detach().cpu().numpy())
plt.title("Human model with h=1.9m", fontsize=35)

ax = fig.add_subplot(224, projection='3d')
h = Human(1.5, "cpu", "h36m")
model_1 = h.update_pose(r)
plot_3d(ax, bones, model_1.detach().cpu().numpy())
plt.title("Human model with h=1.5m", fontsize=35)

plt.show()
