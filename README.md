# COLMAP_SLAM

https://github.com/3DOM-FBK/COLMAP_SLAM/assets/93863149/af549427-8e87-445d-92f3-1c14e42b5d5a

Visual-SLAM based on COLMAP API mainly intended for the development and test of new SLAM features (deep-learning based tie points and matching, keyframe selection, global optimization, etc). The repository uses Kornia (https://github.com/kornia/kornia) for matching, and for now only Key.Net+HardNet8 is implemented. All local features made available by Kornia will be added shortly. Other interest points: RootSIFT from COLMAP, ORB from OpenCV, and ALIKE (https://github.com/Shiaoming/ALIKE).

Currently only the monocular scenario is supported, but we are joining an other repository with other features (multi-cameras, GNSS, IMU). If interested in the project please contact us, you are free to join.

Note the repository is an adaptation of COLMAP to work in real-time, for code and license please refer to: <https://github.com/colmap/colmap>.

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
- [x] Add brute-force kornia matcher
- [ ] Add SuperGlue
- [ ] From Kornia add Loftr, Adalam
- [ ] Join multi camera code
- [ ] Join GNSS positioning
- [ ] Join IMU aiding
- [ ] Divide reconstruction in voxels to optimeze running time (loop closure based on nerest voxel)

### Reference

Code authors: Luca Morelli and Francesco Ioli.

To be added.

### Notes

- In mapper.ini keep transitivity high.
