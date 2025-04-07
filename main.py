import os
import cv2
from tqdm import tqdm
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

    start_frame = config['mapping']['start_frame']
    working_dir = args.work_dir
    frames_dir = working_dir / "images"
    frames_cam0 = os.listdir(frames_dir / "cam0")
    frames_cam0.sort()
    pose_changes = []

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

    for frame_index in tqdm(range(start_frame+1, 50)):
    #for frame_index in tqdm(range(start_frame+1, len(frames_cam0))):
        pose_change = visual_odometry.run(frames_cam0[frame_index])
        pose_changes.append(pose_change)

    out_file = open(out_file_path, "a")
    for i in range(len(pose_changes)):
        try:
            out_file.write(f"{pose_changes[i][0][0]} {pose_changes[i][0][1]} {pose_changes[i][0][2]} {pose_changes[i][0][3]}\n")
        except:
            pass
        #out_images_file = open(self.out_images_file_path, "a")
        #out_file.write("# {new_kfrm.name} {cumulative[0]} {cumulative[1]} {cumulative[2]} {norm[0]} {norm[1]} {norm[2]} {cumulativa_quaternion[0]} {cumulativa_quaternion[1]} {cumulativa_quaternion[2]} {cumulativa_quaternion[3]} {delta_t[0]} {delta_t[1]} {delta_t[2]} {delta_q[0]} {delta_q[1]} {delta_q[2]} {delta_q[3]}\n")
        #out_file.write(f"{new_kfrm.name} {cumulative[0]} {cumulative[1]} {cumulative[2]} {norm[0]} {norm[1]} {norm[2]} {cumulativa_quaternion[0]} {cumulativa_quaternion[1]} {cumulativa_quaternion[2]} {cumulativa_quaternion[3]} {delta_t[0]} {delta_t[1]} {delta_t[2]} {delta_q[0]} {delta_q[1]} {delta_q[2]} {delta_q[3]}\n")
        #out_images_file.write(f"{self.keyframes_names[new_kfrm.name]} {cumulativa_quaternion[0]} {cumulativa_quaternion[1]} {cumulativa_quaternion[2]} {cumulativa_quaternion[3]} {t[0]} {t[1]} {t[2]} 1 {new_kfrm.name}\n\n")
    out_file.close()
        #out_images_file.close()

if __name__ == "__main__":
    main()