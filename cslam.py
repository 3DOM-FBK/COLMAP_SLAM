"""Run colmap-odometry (git submodule in third_party/colmap-odometry).

Example:
    python cslam.py                      # runs on the sample CARLA dataset
    python cslam.py --gui                # same, with a live trajectory plot
    python cslam.py -i path/to/images -w path/to/output
"""

import argparse
import functools
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLMAP_ODOMETRY = ROOT / "third_party" / "colmap-odometry"

DEFAULT_IMAGES = ROOT / "assets" / "sample_carla_dataset"
DEFAULT_CONFIG = ROOT / "config" / "config_carla.yaml"
DEFAULT_CAMERA = ROOT / "calibration" / "calibration_carla.yaml"
DEFAULT_RIG = ROOT / "calibration" / "camera_rig_carla.yaml"
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
    parser.add_argument("--gui", action="store_true",
                        help="Show the estimated trajectory live while running")
    return parser.parse_args()


class TrajectoryViewer:
    """Minimal live top-down (X-Z) plot of the camera trajectory.

    Poses are accumulated from the relative motions returned by
    VisualOdometry.run(), exactly as colmap-odometry's main.py does when it
    writes trajectory.txt.
    """

    def __init__(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from pyquaternion import Quaternion

        self.plt = plt
        self.t = np.zeros(3)
        self.q = Quaternion()
        self.xs, self.zs = [], []

        plt.ion()
        self.fig, self.ax = plt.subplots(num="cslam - trajectory")
        (self.line,) = self.ax.plot([], [], "-", color="tab:blue", lw=1.5, label="trajectory")
        (self.head,) = self.ax.plot([], [], "o", color="tab:red", label="current pose")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Z [m]")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left")
        plt.show(block=False)

    def update(self, pose_change) -> None:
        try:
            image, _, delta_t, delta_q, _, _ = pose_change[0]
        except (IndexError, TypeError, ValueError):
            return
        if delta_t is None or delta_q is None:  # frame not localized
            return

        self.t = self.t + self.q.inverse.rotate(delta_t)
        self.q = delta_q * self.q
        self.xs.append(self.t[0])
        self.zs.append(self.t[2])

        if not self.plt.fignum_exists(self.fig.number):  # window closed by user
            return
        self.line.set_data(self.xs, self.zs)
        self.head.set_data([self.xs[-1]], [self.zs[-1]])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f"{image}  -  {len(self.xs)} poses")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def keep_open(self) -> None:
        """Block until the user closes the window."""
        if self.plt.fignum_exists(self.fig.number):
            self.ax.set_title(f"Finished  -  {len(self.xs)} poses")
            self.plt.ioff()
            self.plt.show()


def attach_viewer(viewer: TrajectoryViewer) -> None:
    """Wrap VisualOdometry.run so that each estimated pose is sent to the viewer."""
    from src.odometry.odometry import VisualOdometry

    original_run = VisualOdometry.run

    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        pose_change, log = original_run(self, *args, **kwargs)
        viewer.update(pose_change)
        return pose_change, log

    VisualOdometry.run = run


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

    viewer = None
    if args.gui:
        viewer = TrajectoryViewer()
        attach_viewer(viewer)

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

    if viewer is not None:
        viewer.keep_open()


if __name__ == "__main__":
    main()
