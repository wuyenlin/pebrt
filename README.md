## PETR (Pose Estimation using TRansformer)

This repository contains files used in the thesis done by Yen-Lin Wu in partial fulfillment of his MSc programme in Delft University of Technology (2021), supervised by Osama Mazhar and Jens Kober. 

The thesis aims to address the widely challenged computer vision task - Human Pose Estimation. 
Different from most existing methods, we propose a novel estimating technique that discards convolutional layers, using only Transformer layers.
On top of that, we integrate human anthropometric constraints to improve prediction accuracies and achieved real-time 3D human pose estimation.


## Getting started

Make sure there is 200GB on local hard drive in order to save the original dataset and its processed frames.

### Installation

```
git clone https://github.com/wuyenlin/thesis
cd thesis/
pip3 install -r requirements.txt
```

### Download dataset

The dataset MPI-INF-3DHP used in this project can be downloaded [here](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).

After downloading the dataset, run the following command to extract frames:
```
python3 utils/File.py
python3 merge.py
```

