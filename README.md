# COLMAP_SLAM


- NEW BRANCH UNDER DEVELOPMENT -


https://github.com/3DOM-FBK/COLMAP_SLAM/assets/93863149/af549427-8e87-445d-92f3-1c14e42b5d5a

COLMAP_SLAM is a Visual-SLAM based on COLMAP API and is mainly intended for the development and testing of new SLAM features (deep-learning based tie points and matching, keyframe selection, global optimization, etc). The repository relys on Kornia (https://github.com/kornia/kornia) for matching, and some detectors and descriptors.

Monocular and multicamera scenario are supported. We are joining an other repository with other features (loop-closure detection, GNSS and IMU aiding). We are optimizing the code for very long sequences. If interested in the project please contact us, you are free to join.

Note the repository is an adaptation of COLMAP to work in real-time, for code and license please refer to [COLMAP](https://github.com/colmap/colmap).

| Local Feature      | Supported | Matcher     | Supported |
|----------          |---------- |----------   |---------- |
| RootSIFT           | yes       | Brute Force | yes       |
| ORB                | yes       | SuperGlue   | yes       |
| Key.Net + HardNet8 | yes       | LightGlue   | yes       |
| ALIKE              | yes       | LoFTR       | not yet   |
| ALIKED             | not yet   | ADALAM      | yes       |
| SuperPoint         | yes       | ...         |           |
| DISK               | yes       | ...         |           |
| DeDoDe             | not yet   | ...         |           |
| DKM                | not yet   | ...         |           |


## EuRoC

Download a dataset from [https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and unzip it in the `raw_data` folder. For instance, for Machine Hall 01 use only cam0.

In linux, you can use the following commands:

```bash
mkdir raw_data
wget http://robotics.ethz.ch/\~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip -P raw_data
unzip -q raw_data/MH_01_easy.zip -d raw_data/MH_01_easy
```

## Install and run
### Install in Conda Environment

To install COLMAP_SLAM in a conda environment (Linux, Windows, MacOS)

```bash
conda create -n colmap_slam python=3.9.17
conda activate colmap_slam
```
Install pytorch. See [https://pytorch.org/get-started/locally/#linux-pip](https://pytorch.org/get-started/locally/#linux-pip)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install COLMAP
If you have COLMAP installed yet on your PC, remeber to set the path to the COLMAP executable in `config.ini`, e.g., for linux:

```bash
COLMAP_EXE_DIR = /usr/local/bin/
```
For the installation of COLMAP see the official documentation https://colmap.github.io/install.html
You can find some example in the following section Usage Examples

### Run COLMAP_SLAM

Change `config.ini` according to your needs and run

```bash
python3 main.py
```

### Options

There are some options to change in `config.ini` to run COLMAP_SLAM:
If you are simulating a real time process you have to set

```USE_SERVER = False
SIMULATOR_IMG_DIR = path/to/frame/folder # for instance "./raw_data/MH_01_easy/mav0/cam0/data"
STEP = 5 # If you have high frame rate you can skip some images to increase speed
SIMULATOR_SLEEP_TIME = 0.25 # Time to wait to copy next image
IMG_FORMAT = jpg

COLMAP_EXE_DIR = /usr/local/bin/
```

This should be enough to run with default options. All the options related to the real time processing can be found in config.ini.
The options related to COLMAP APIs are in the lib folder with format .ini


### TODO

- [x] Add full compatibility to Kornia local features
- [x] Join multi camera code
- [ ] Join loop-closure detection
- [ ] Join GNSS positioning
- [ ] Join IMU aiding
- [ ] Pycolmap
- [ ] Optimize the code for long sequences
- [ ] Add option to resize too big images
- [ ] Add pose graph optimization

### Reference

Morelli, L., Ioli, F., Beber, R., Menna, F., Remondino, F. and Vitti, A., 2023. COLMAP-SLAM: A FRAMEWORK FOR VISUAL ODOMETRY. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48, pp.317-324.

### Acknowledgements

For keyframe selection and display: ALIKE (https://github.com/Shiaoming/ALIKE)
For detectors and matching: Kornia (https://github.com/kornia/kornia)


### License

MIT License

Copyright (c) 2023 3DOM-FBK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in alls
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.