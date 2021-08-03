### Download dataset

### 1. MPI-INF-3DHP
The MPI-INF-3DHP dataset can be downloaded [here](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).

After downloading the dataset, run the following command to extract frames and create `.npz` files for each character:
```
python3 common/mpi_dataset.py
```

### 2. Human3.6M
1. Download the Human3.6M dataset from their [official website](vision.imar.ro/human3.6m/) and follow the setup from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) to create `data_2d_h36m_gt.npz` and `data_3d_h36m.npz`.

or

2. Download 2D detection files for Human3.6M from the same [page](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).

After downloading the files, your directory should look like this:
```
```
Now, run the following command to extract video frames and merge 2D/3D annotations into one single `.npz` file.
```
python3 common/h36m_dataset.py
```
