# One pose fits all

This repository contains files used in the thesis done by Yen-Lin Wu in partial fulfillment of his MSc programme in Mechanical Engineering at Delft University of Technology (2021), supervised by Osama Mazhar and Jens Kober. 

The thesis aims to address the widely challenged computer vision task - 3D Human Pose Estimation. 
Different from most existing methods, we propose a novel estimating technique that discards convolutional layers, using only Transformer layers.
On top of that, we integrate human kinemtic constraints to improve prediction accuracies and proposed a new evaluation metric that focuses on human postures, independent of human body shape, age, or gender.

<p align="center"><img src="doc/output.gif" width="55%" alt="" /></p>

## PEBRT (Pose Estimation by Bone Rotation using Transformer)

PEBRT estimates rotation matrix parameters for each bone which are applied to a human kinematic model.
Each rotation matrix is recovered by Gram-Schmidt orthogonalization proposed by [Zhou et al.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf).



### Installation
Clone the repository and install required dependencies to proceed.
```
git clone https://github.com/wuyenlin/thesis
cd thesis/
pip3 install -r requirements.txt
```
For dataset setup, please refer to [`DATASETS.md`](DATASETS.md).


### Evaluation our pre-trained models
Download pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1OYqnEO28A0Ft5XAw4YeBzkK9NNOknZqh?usp=sharing).
For example, to run evaluation on 4 layers of Transformer Encoders:
```
python3 lift.py --num_layers 4 --eval --checkpoint ./all_4_lay_epoch_latest.bin
```


## Training from scratch
To start training the model with 1 layer of Transformer Encoder, run
```
python3 lift.py --num_layers 1
```

If you are running on a SLI enabled machine or computing cluster, run the following Pytorch DDP code (example of using 2 GPUs):
```
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 lift.py
```


### TODO
- [x] Animate results (see animation.py)
- [x] Create evaluation metrics for bone rotation error
- [x] Add kinematic constraints
- [x] Train and test on Human3.6M
- [x] Run on distributed systems (for SLI)
- [x] Added human model for both Human3.6M and MPI-IND-3DHP datasets
- [x] Fix camera angle issue / add 3D joint position in loss 
- [x] Test evaluation metrics on existing methods (working on it now)
- [ ] Separate human model configurations into yaml files
- [ ] Online implementations of PETR (training & finetuning now)