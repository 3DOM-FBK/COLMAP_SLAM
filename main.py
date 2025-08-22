import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.odometry.odometry import VisualOdometry

REINIZIALIZE_AFTER = -1  # Reinitialize after this many frames, -1 disables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=Path, required=True)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-a", "--camera", type=Path, required=True)
    parser.add_argument("-w", "--work_dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.camera) as f:
        camera_config = yaml.safe_load(f)

    start_frame, end_frame = config['general']['frames_range']
    working_dir = args.work_dir
    frames_dir = args.images
    cam0_dir = frames_dir / "cam0"

    # Use sorted Path.glob for better performance
    frames_cam0 = sorted([f.name for f in cam0_dir.iterdir() if f.is_file()])
    if end_frame == -1:
        end_frame = len(frames_cam0)

    # Prepare output files
    out_file_path = working_dir / "trajectory.txt"
    out_images_file_path = working_dir / "images.txt"
    for path in [out_file_path, out_images_file_path]:
        if path.exists():
            path.unlink()

    visual_odometry = VisualOdometry(
        working_dir=working_dir,
        config=config,
        camera_config=camera_config,
    )

    pose_changes = []

    # Precompute reinitialization frame set for O(1) check
    reinit_set = {REINIZIALIZE_AFTER} if REINIZIALIZE_AFTER >= 0 else set()

    for frame_index in tqdm(range(start_frame + 1, end_frame)):
        img_name = frames_cam0[frame_index]
        images = []

        for cam in visual_odometry.cameras:
            img_path = frames_dir / cam / img_name
            # Read in color directly, skip conversion if not required
            cv_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            images.append(cv_img)

        reinitialize = frame_index in reinit_set
        pose_change, log = visual_odometry.run(img_name, images, reinitialize=reinitialize)
        pose_changes.append(pose_change)

        if config['general']['log']:
            print(log)

    summary = visual_odometry.get_performance_summary()
    print(summary)

    # Write output in bulk to improve speed
    traj_lines = []
    img_lines = []

    for changes in pose_changes:
        try:
            image, id, delta_t, delta_q, t_cumulative, q_cumulative = changes[0]
            t_ = -q_cumulative.rotation_matrix @ t_cumulative
            traj_lines.append(f"{image} {t_cumulative[0]} {t_cumulative[1]} {t_cumulative[2]}\n")
            img_lines.append(f"{id} {q_cumulative[0]} {q_cumulative[1]} {q_cumulative[2]} {q_cumulative[3]} "
                             f"{t_[0]} {t_[1]} {t_[2]} 1 {image}\n\n")
        except Exception:
            continue

    out_file_path.write_text("".join(traj_lines))
    out_images_file_path.write_text("".join(img_lines))

if __name__ == "__main__":
    main()
