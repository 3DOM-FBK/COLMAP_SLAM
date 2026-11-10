import os
import shutil
import argparse

from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare TartanAir dataset.")
    parser.add_argument(
        "--work-dir", "-w",
        help="Path to the dataset work directory",
        type=Path,
    )
    parser.add_argument(
        "--out-dir", "-o",
        help="Path to the dataset output directory",
        type=Path,
    )
    args = parser.parse_args()

    work_dir = args.work_dir

    path_image_lcam_left = work_dir / "image_lcam_left"
    path_image_lcam_front = work_dir / "image_lcam_front"
    path_image_lcam_right = work_dir / "image_lcam_right"
    path_image_rcam_front = work_dir / "image_rcam_front"
    path_image_rcam_right = work_dir / "image_rcam_right"

    for path in [path_image_lcam_left, path_image_lcam_front, path_image_lcam_right, path_image_rcam_front, path_image_rcam_right]:
        image_path = path / "ArchVizTinyHouseDay" / "Data_easy" / "P000" / path.name
        images = os.listdir(image_path)

        out_path = args.out_dir / path.name
        out_path.mkdir(parents=True, exist_ok=True)

        for img in images:
            new_img_name = img.split("_")[0] + ".png"
            src = image_path / img
            dst = out_path / new_img_name
            shutil.copy(src, dst)