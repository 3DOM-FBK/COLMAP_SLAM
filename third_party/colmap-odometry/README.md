# Colmap-Odometry

Colmap-Odometry is a visual odometry framework based on pycolmap and is mainly intended for the development and testing of new VO/SLAM features (deep-learning based tie points and matching, keyframe selection, global optimization, etc).

Feel free to join the project!

Key fratures:
* completly build on pycolmap, few dependency and easy to install
* windowed bundle adjustement
* monocular camera supported
* stereo cameras supported
* multi-camera supported (requires partial FOV overlap)
* build for long sequences
* deep learning based features support


## Installation
For installing colmap-odometry, we recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then, you can install colmap-odometry using uv:

```bash
uv pip install -e .
```

This command will install the package in editable mode, allowing you to modify the source code and see changes immediately without needing to reinstall. If you want to use colmap-slam as a non-editable library, you can also install it without the `-e` flag.

Check that cuda is available in `pytorch`, or manually install pytorch from https://pytorch.org/ to have GPU support.

## Running the code

```
python ./main.py -c ./config/config_carla.yaml -a ./calibration/calibration_carla.yaml -r ./calibration/camera_rig_carla.yaml -i ./assets/sample_carla_dataset -w ./assets/output
```

The full trajectory is stored in the output folder: `trajectory.txt` in world reference system or `images.txt` following COLMAP conventions.

If you want to run on the euroc dataset you can download the data from https://projects.asl.ethz.ch/datasets/euroc-mav/ and run:

```
python ./main.py -c ./config/config_euroc.yaml -a ./calibration/calibration_euroc.yaml -r ./calibration/camera_rig_euroc.yaml -i path/to/image/folder -w ./path/to/output/folder
```

### Reference

```bibtex
@article{morelli2025deep,
  title={Deep Learning in Visual Odometry for Autonomous Driving},
  author={Morelli, Luca and Tryba{\l}a, Pawe{\l} and Razzino, Armando Vittorio and Remondino, Fabio},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={93--99},
  year={2025},
  publisher={Copernicus GmbH}
}
```

```bibtex
@article{morelli2023colmap,
  title={COLMAP-SLAM: A framework for visual odometry},
  author={Morelli, Luca and Ioli, Francesco and Beber, Raniero and Menna, Fabio and Remondino, Fabio and Vitti, Alfonso},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={317--324},
  year={2023},
  publisher={Copernicus Publications G{\"o}ttingen, Germany}
}
```
