import os
import yaml
import argparse

from pathlib import Path
from src.odometry.odometry import VisualOdometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Path to general configuration file", required=True)
    parser.add_argument("-a", "--camera", type=Path, help="Path to camera configuration file", required=True)
    parser.add_argument("-w", "--work_dir", type=Path, help="Path to the working directory", required=True)
    args = parser.parse_args()

    config_yaml = args.config
    with open(config_yaml) as config_yaml:
        config = yaml.safe_load(config_yaml)
    
    camera_yaml = args.camera
    with open(camera_yaml) as camera_yaml:
        camera_config = yaml.safe_load(camera_yaml)
    
    visual_odometry = VisualOdometry(
        working_dir=args.work_dir,
        config=config,
        camera_config = camera_config,
    )

    visual_odometry.run()

if __name__ == "__main__":
    main()