"""Run colmap-odometry (git submodule in third_party/colmap-odometry).

Example:
    python cslam.py                      # runs on the sample CARLA dataset
    python cslam.py -i path/to/images -w path/to/output
"""

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLMAP_ODOMETRY = ROOT / "third_party" / "colmap-odometry"

DEFAULT_IMAGES = COLMAP_ODOMETRY / "assets" / "sample_carla_dataset"
DEFAULT_CONFIG = COLMAP_ODOMETRY / "config" / "config_carla.yaml"
DEFAULT_CAMERA = COLMAP_ODOMETRY / "calibration" / "calibration_carla.yaml"
DEFAULT_RIG = COLMAP_ODOMETRY / "calibration" / "camera_rig_carla.yaml"
DEFAULT_WORK_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run colmap-odometry visual odometry on an image sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--images", type=Path, default=DEFAULT_IMAGES,
                        help="Folder with one subfolder per camera (cam0, cam1, ...)")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                        help="Odometry config file (.yaml)")
    parser.add_argument("-a", "--camera", type=Path, default=DEFAULT_CAMERA,
                        help="Camera calibration file (.yaml)")
    parser.add_argument("-r", "--rig", type=Path, default=DEFAULT_RIG,
                        help="Camera rig file (.yaml)")
    parser.add_argument("-w", "--work_dir", type=Path, default=DEFAULT_WORK_DIR,
                        help="Output folder (trajectory.txt, images.txt, database.db)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_script = COLMAP_ODOMETRY / "main.py"
    if not main_script.exists():
        sys.exit(
            f"colmap-odometry not found in {COLMAP_ODOMETRY}.\n"
            "Run: git submodule update --init --recursive"
        )

    for name in ("images", "config", "camera", "rig"):
        path = getattr(args, name)
        if not path.exists():
            sys.exit(f"--{name} not found: {path}")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # colmap-odometry imports its modules as `src.odometry...`
    sys.path.insert(0, str(COLMAP_ODOMETRY))

    # Forward the arguments to the upstream main.py and run it as a script
    sys.argv = [
        str(main_script),
        "-i", str(args.images.resolve()),
        "-c", str(args.config.resolve()),
        "-a", str(args.camera.resolve()),
        "-r", str(args.rig.resolve()),
        "-w", str(args.work_dir.resolve()),
    ]
    runpy.run_path(str(main_script), run_name="__main__")


if __name__ == "__main__":
    main()
