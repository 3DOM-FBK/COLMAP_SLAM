import os
import cv2
import yaml
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from COLMAP_SLAM.src.odometry.odometry import VisualOdometry

class CSLAM():
    def __init__(self):
        config_yaml =  "/home/carla/thirdparty/COLMAP_SLAM/config/config_carla.yaml"
        with open(config_yaml) as config_yaml:
            config = yaml.safe_load(config_yaml)
        camera_yaml = "/home/carla/thirdparty/COLMAP_SLAM/calibration/calibration_carla.yaml"
        with open(camera_yaml) as camera_yaml:
            camera_config = yaml.safe_load(camera_yaml)
        self.pose_changes = []
        self.last_info = None
        self.processing = False
        wdir = Path("/home/carla/thirdparty/COLMAP_SLAM/temp")
        rig_yaml = Path("/home/carla/thirdparty/COLMAP_SLAM/calibration/camera_rig_carla.yaml")
        with open(rig_yaml) as rig_yaml:
            rig_config = yaml.safe_load(rig_yaml)
        if os.path.exists(wdir): [shutil.rmtree(p) if os.path.isdir(p := os.path.join(wdir, f)) else os.remove(p) for f in os.listdir(wdir)]
        self.visual_odometry = VisualOdometry(
            working_dir=wdir,
            config=config,
            camera_config = camera_config,
            rig_config=rig_config
        )

    def process_stereopair(self, frame, im0, im1, reinitialize=False):
        self.processing = True
        pose_change, info = self.visual_odometry.run(frame, (im0, im1), reinitialize)
        self.pose_changes.append(pose_change)
        self.last_info = info
        self.processing = False

    def get_last_pose_change(self):
        if len(self.pose_changes) > 0 and not self.processing:
            return(self.pose_changes[-1])
        else:
            return None

    def get_last_info(self):
        if self.last_info is not None and not self.processing:
            return(self.last_info)
        else:
            return None
