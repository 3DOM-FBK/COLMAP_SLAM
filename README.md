# COLMAP_SLAM

SLAM based on COLMAP API for both Windows and Linux OS. The repository is under construction, if interested in the project you are free to join. Please note the repository is an adaptation of COLMAP to work in real-time. For code and license please refer to: <https://github.com/colmap/colmap>.

## EuRoC

Download dataset from <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
For instance, for Machine Hall 01 use only cam0.

### Install in Conda Environment

To install a an Anaconda environment (Linux, Windows, MacOS)

```bash
conda create -n colmap_slam python=3.10
conda activate colmap_slam
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Remember to install pytorch

See https://pytorch.org/get-started/locally/#linux-pip

### Run COLMAP_SLAM

Change `conf.ini` according to your needs and run

```bash
python3 main.py
```

### TODO

- [x] Add full compatibility to Kornia local features
- [ ] Add kornia matcher (use lib/import_local_features.py)
- [ ] Use always logger instead of print
- [ ] Divide reconstruction in voxels to optimeze running time (loop closure based on nerest voxel)
- [ ] Join multi camera code
- [ ] Join GNSS positioning
- [ ] Join IMU aiding

### Reference

To be added.
