# COLMAP_SLAM

https://github.com/3DOM-FBK/COLMAP_SLAM/assets/93863149/af549427-8e87-445d-92f3-1c14e42b5d5a

Visual-SLAM is based on COLMAP API and is mainly intended for the development and testing of new SLAM features (deep-learning based tie points and matching, keyframe selection, global optimization, etc). The repository uses Kornia (https://github.com/kornia/kornia) for matching, and for now only Key.Net+HardNet8 is implemented. All local features made available by Kornia will be added shortly. Other interest points: RootSIFT from COLMAP, ORB from OpenCV, and ALIKE (https://github.com/Shiaoming/ALIKE).

Currently only the monocular scenario is supported, but we are joining an other repository with other features (multi-cameras, GNSS, IMU). If interested in the project please contact us, you are free to join.

Note the repository is an adaptation of COLMAP to work in real-time, for code and license please refer to: <https://github.com/colmap/colmap>.

## EuRoC

Download dataset from <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
For instance, for Machine Hall 01 use only cam0.

### Install in Conda Environment

To install COLMAP_SLAM in a conda environment (Linux, Windows, MacOS)

```bash
conda create -n colmap_slam python=3.10
conda activate colmap_slam
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Remember to install pytorch

See https://pytorch.org/get-started/locally/#linux-pip

### Run COLMAP_SLAM

Change `config.ini` according to your needs and run

```bash
python3 main.py
```


### Usage Example in LINUX [Ubuntu 22.04 TESTED]

1) Clone the repo to your local folder 
```
git clone https://github.com/3DOM-FBK/COLMAP_SLAM.git
```

2) Enter the cloned repo
```
cd COLMAP_SLAM
```

3) Create the conda env with Python 3.10 and enter into 'colmap_slam'
``` 
conda create -n colmap_slam python=3.10
conda activate colmap_slam
```

4) Upgrade pip
```
python -m pip install --upgrade pip
```

5) Allow for cuda capabilities check [remove old key] 
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb ~3.02GB

```
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

IF GDS is of interest
```
sudo apt-get install nvidia-gds
```

**WARNING** - reboot needed - **WARNING**

rememnre to re-activate the conda env after reboot

``` 
sudo reboot
```

update system variables
```
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

verify driver versions

```
cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  530.30.02  Wed Feb 22 04:11:39 UTC 2023
GCC version:  gcc version 11.3.0 (Ubuntu 11.3.0-1ubuntu1~22.04.1) 
```

testing CUDA

```
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j 8
```

then check if one example is working

```
cd Samples/1_Utilities/deviceQuery
make clean
make -j 8
./deviceQuery

>>> ./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA RTX A3000 12GB Laptop GPU"
  CUDA Driver Version / Runtime Version          12.1 / 12.1
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12045 MBytes (12630294528 bytes)
  (032) Multiprocessors, (128) CUDA Cores/MP:    4096 CUDA Cores
  GPU Max Clock rate:                            1545 MHz (1.54 GHz)
  Memory Clock rate:                             7001 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 3145728 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.1, CUDA Runtime Version = 12.1, NumDevs = 1
Result = PASS
```

compare your output with: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#running-binaries-valid-results-from-sample-cuda-devicequery-program


install 3rd party libraries
```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev
```

6) Install pyTorch (with CUDA) use the https://pytorch.org/get-started/locally/
```
pip3 install torch torchvision torchaudio
```

7) Install colmap based on your system: https://colmap.github.io/install.html#pre-built-binaries
```
apt list | grep colmap
>>colmap/jammy 3.7-2 amd64
```

```
sudo apt install colmap
```

8) Install requirements 
```
pip install -r requirements.txt
```
9) Download EuRoC dataset https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
MH_01_easy.zip ~1.5 GB

```
mkdir raw_data
cd raw_data
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
``` 

10) place downloaded dataset (e.g. **Machine Hall 01** in a folder) 
```
mv ~/Downloads/MH_01_easy.zip ~/COLMAP_SLAM/raw_data/MH_01_easy.zip
unzip -d ./MH_01_easy/ MH_01_easy.zip
```

11) Set up the **config.ini** file with proper location of dataset
```
SIMULATOR_IMG_DIR = ~/COLMAP_SLAM/raw_data/MH_01_easy/mav0/cam0/data
COLMAP_EXE_DIR = /usr/bin/
```

12) Run COLMAP_SLAM
```
python3 main.py
```

Currently some issues with pop-out window to be fixed

### Usage Example in WSL2 [to be TESTED]
1) Clone the repo to your local folder 
``` git clone https://github.com/3DOM-FBK/COLMAP_SLAM.git  ```
2) Enter the cloned repo
``` cd COLMAP_SLAM```
3) Create the conda env with Python 3.10 and enter into 'colmap_slam'
``` conda create -n colmap_slam python=3.10
conda activate colmap_slam
```
4) Upgrade pip ```python -m pip install --upgrade pip```
5) Allow WSL2 for cuda capabilities check 
```
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
```
cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb ~2.6GB
```
sudo apt-key del 7fa2af80
sudo apt install nvidia-cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# Can be skipped if altready downloaded
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb

# TO BE VERIFIED
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb 
sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
check cuda version
```nvcc --version```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

6a) Install pyTorch (with CUDA) use the https://pytorch.org/get-started/locally/
```
pip3 install torch torchvision torchaudio
```


6b) Install pyTorch (no CUDA) [not much of use]
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
7) Install colmap based on your system: https://colmap.github.io/install.html#pre-built-binaries
```
apt list | grep colmap
>>colmap/focal 3.6+dev2+git20191105-1build1 amd64
```
```
sudo apt install colmap
```
8) Install requirements ```pip install -r requirements.txt```
9) Download EuRoC dataset https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
MH_01_easy.zip ~1.5 GB
```http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip``` 
10) place downloaded dataset (e.g. **Machine Hall 01** in a folder) 
```
mv ~/Downloads/MH_01_easy.zip ~/COLMAP_SLAM/raw_data/MH_01_easy.zip
unzip -d ./MH_01_easy/ MH_01_easy.zip
```
11) Set up the **config.ini** file with proper location of dataset
```
SIMULATOR_IMG_DIR = ~/COLMAP_SLAM/raw_data/MH_01_easy/mav0/cam0/data

```
12) Run 
```
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
- [ ] Example testing
### Reference

Code authors: Luca Morelli and Francesco Ioli.

Reference Article:
COLMAP-SLAM: A FRAMEWORK FOR VISUAL ODOMETRY.

Authors: 
L. Morelli, F. Ioli, R. Beber, F. Menna, F. Remondino, A. Vitti

### Notes

- In mapper.ini keep transitivity high.

