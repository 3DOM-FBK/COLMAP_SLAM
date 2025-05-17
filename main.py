import os
import cv2
import yaml
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from src.odometry.odometry import VisualOdometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=Path, help="Path to images directory", required=True)
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

    start_frame, end_frame = config['general']['frames_range']
    working_dir = args.work_dir
    frames_dir = args.images
    frames_cam0 = os.listdir(frames_dir / "cam0")
    frames_cam0.sort()
    pose_changes = []
    if end_frame == -1:
        end_frame = len(frames_cam0)

    out_file_path = working_dir / "trajectory.txt"
    if out_file_path.exists():
        out_file_path.unlink()

    out_images_file_path = working_dir / "images.txt"
    if out_images_file_path.exists():
        out_images_file_path.unlink()

    ## Visualize cam0 frames
    #for image_file in frames_cam0:
    #    image_path = str(frames_dir / 'cam0' / image_file)
    #    image = cv2.imread(image_path)
    #    cv2.imshow("Image", image)
    #    cv2.waitKey(1)
    #cv2.destroyAllWindows()

    visual_odometry = VisualOdometry(
        working_dir=working_dir,
        config=config,
        camera_config = camera_config,
    )

    for frame_index in tqdm(range(start_frame+1, end_frame)):
        img = frames_cam0[frame_index]
        images = []
        for c in visual_odometry.cameras:
            cv2_img = cv2.imread(str(frames_dir / c / img))
            img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        pose_change = visual_odometry.run(frames_cam0[frame_index], images)
        pose_changes.append(pose_change)

    out_file = open(out_file_path, "a")
    out_images_file = open(out_images_file_path, "a")
    for i in range(len(pose_changes)):
        try:
            image, id, delta_t, delta_q, t_cumulative, q_cumulative = pose_changes[i][0]
            norm = q_cumulative.inverse.rotate(np.array([0, 0, 1]))
            t_ = -q_cumulative.rotation_matrix @ t_cumulative
            out_file.write(f"{image} {t_cumulative[0]} {t_cumulative[1]} {t_cumulative[2]}\n")
            out_images_file.write(f"{id} {q_cumulative[0]} {q_cumulative[1]} {q_cumulative[2]} {q_cumulative[3]} {t_[0]} {t_[1]} {t_[2]} 1 {image}\n\n")
        except:
            pass

    out_file.close()
    out_images_file.close()

if __name__ == "__main__":
    main()