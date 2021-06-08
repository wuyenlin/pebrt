# Thesis proejct (Q3/Q4 2021)

This repository contains files used in the thesis done by Yen-Lin Wu in partial fulfillment of his MSc programme in Mechanical Engineering at Delft University of Technology (2021), supervised by Osama Mazhar and Jens Kober. 

The thesis aims to address the widely challenged computer vision task - Human Pose Estimation. 
Different from most existing methods, we propose a novel estimating technique that discards convolutional layers, using only Transformer layers.
On top of that, we integrate human kinemtic constraints to improve prediction accuracies and proposed a new evaluation metric that focuses on human postures, independent of human body shape, age, or gender.


## PEBRT (Pose Estimation by Bone Rotation using Transformer)

PEBRT estimates rotation matrix parameters for each bone that are recovered by Gram-Schmidt orthogonalization.

![](doc/output.gif)

## PETR (Pose Estimation using TRansformer)

PETR is an end-to-end lifting pipeline used to predict human 3D keypoints from RGB images.

![](doc/demo_1.png)

## Getting started

Make sure there is 200GB on local hard drive in order to save the original dataset and its processed frames.

### Installation

```
git clone https://github.com/wuyenlin/thesis
cd thesis/
pip3 install -r requirements.txt
```

### Download dataset

The MPI-INF-3DHP dataset can be downloaded [here](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).

After downloading the dataset, run the following command to extract frames and create npz files for dataloader:
```
python3 common/mpi_dataset.py
```

- [x] Animate results (see animation.py)
- [x] Create evaluation metrics for bone rotation error
- [ ] Fix camera angle issue
- [ ] Add kinematic constraints
- [ ] Online implementations of PEBRT
- [ ] Test evaluation metrics on existing methods
- [ ] Run on distributed systems (for SLI)