## Download dataset

### 1. Human3.6M
Download the Human3.6M dataset from their [official website](vision.imar.ro/human3.6m/) and follow the setup from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) to create `data_2d_h36m_gt.npz` and `data_3d_h36m.npz`.

Create a `h36m/` folder and place the `.npz` files inside.

Now, run the following command to extract video frames and merge 2D/3D annotations into one single `.npz` file.
```
python3 common/h36m_dataset.py
```
This will generate `data_h36m_frame_all.npz` under the `h36m/` folder.

Finally, your folder should look like this.
```
animation/
common/
h36m/
├── data_2d_g36m_gt.npz
├── data_3d_h36m.npz
└── data_h36m_frame_all.npz
doc/
```


### 2. MPI-INF-3DHP
The MPI-INF-3DHP dataset can be downloaded [here](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).

After downloading the dataset, run the following command to extract frames and create `.npz` files for each character:
```
python3 common/mpi_dataset.py
```
(To be finished...)