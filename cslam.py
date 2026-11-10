"""Run colmap-odometry (git submodule in third_party/colmap-odometry).

Example:
    python cslam.py                      # runs on the sample CARLA dataset
    python cslam.py --gui                # same, with a live trajectory plot
    python cslam.py --gnss assets/gnss.txt   # + pose-graph optimization with GNSS
    python cslam.py -i path/to/images -w path/to/output
"""

import argparse
import functools
import runpy
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from trajectory_io import write_trajectory

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
    parser.add_argument("-g", "--gnss", type=Path, default=None,
                        help="GNSS file (lines: image_name x y z). If given, the trajectory "
                             "is refined at the end with a gtsam pose-graph optimization")
    parser.add_argument("--gnss_sigma", type=float, default=0.05,
                        help="GNSS position standard deviation [m]")
    parser.add_argument("--odom_sigma_rot", type=float, default=0.01,
                        help="Odometry relative rotation standard deviation [rad]")
    parser.add_argument("--odom_sigma_trans", type=float, default=0.05,
                        help="Odometry relative translation standard deviation [m]")
    return parser.parse_args()


class OdometryTrajectory:
    """Chains the relative poses returned by VisualOdometry.run() into camera-to-world poses.

    Uses the same accumulation as colmap-odometry's main.py when it writes trajectory.txt.
    """

    def __init__(self):
        self.t = np.zeros(3)   # camera center in world
        self.q = Quaternion()  # world-to-camera rotation
        self.names, self.ids, self.poses_wc, self.covariances = [], [], [], []

    def add(self, pose_change, log) -> bool:
        """Add a frame; returns False if the frame was not localized."""
        try:
            image, image_id, delta_t, delta_q, _, _ = pose_change[0]
        except (IndexError, TypeError, ValueError):
            return False
        if delta_t is None or delta_q is None:
            return False

        self.t = self.t + self.q.inverse.rotate(delta_t)
        self.q = delta_q * self.q
        T = np.eye(4)
        T[:3, :3] = self.q.rotation_matrix.T
        T[:3, 3] = self.t
        self.names.append(image)
        self.ids.append(image_id)
        self.poses_wc.append(T)
        self.covariances.append(self._translation_std(log))
        return True

    @staticmethod
    def _translation_std(log) -> float:
        """Same `cov` value as upstream trajectory.txt: norm of the translation std devs (999 if unknown)."""
        try:
            cov = log["current_frame"]["covariance_matrix_angles_translations"]
        except (KeyError, TypeError):
            cov = None
        if cov is None:
            return 999
        return float(np.linalg.norm(np.sqrt(np.diag(cov)[-3:])))

    def centers(self) -> np.ndarray:
        return np.array([T[:3, 3] for T in self.poses_wc]).reshape(-1, 3)


class TrajectoryViewer:
    """Minimal live top-down (X-Z) plot of the camera trajectory."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(num="cslam - trajectory")
        (self.line,) = self.ax.plot([], [], "-", color="tab:blue", lw=1.5, label="odometry")
        (self.head,) = self.ax.plot([], [], "o", color="tab:red", label="current pose")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Z [m]")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left")
        plt.show(block=False)

    def _is_open(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def _redraw(self, title: str) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, trajectory: OdometryTrajectory) -> None:
        if not self._is_open():  # window closed by user
            return
        c = trajectory.centers()
        self.line.set_data(c[:, 0], c[:, 2])
        self.head.set_data([c[-1, 0]], [c[-1, 2]])
        self._redraw(f"{trajectory.names[-1]}  -  {len(c)} poses")

    def show_optimized(self, optimized: np.ndarray, gnss: np.ndarray) -> None:
        if not self._is_open():
            return
        self.ax.plot(gnss[:, 0], gnss[:, 2], "x", color="tab:green", label="GNSS")
        self.ax.plot(optimized[:, 0], optimized[:, 2], "-", color="tab:orange", lw=1.5,
                     label="optimized")
        self.ax.legend(loc="upper left")
        self._redraw(self.ax.get_title())

    def keep_open(self) -> None:
        """Block until the user closes the window."""
        if self._is_open():
            self.ax.set_title(f"Finished  -  {len(self.line.get_xdata())} poses")
            self.plt.ioff()
            self.plt.show()


def attach(trajectory: OdometryTrajectory, viewer: TrajectoryViewer = None) -> None:
    """Wrap VisualOdometry.run to record each estimated pose (and show it in the viewer)."""
    from src.odometry.odometry import VisualOdometry

    original_run = VisualOdometry.run

    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        pose_change, log = original_run(self, *args, **kwargs)
        if trajectory.add(pose_change, log) and viewer is not None:
            viewer.update(trajectory)
        return pose_change, log

    VisualOdometry.run = run


def optimize_with_gnss(args, trajectory: OdometryTrajectory, viewer: TrajectoryViewer = None) -> None:
    from pose_graph import optimize_pose_graph, read_gnss

    if not trajectory.poses_wc:
        print("Pose graph skipped: no localized frames")
        return
    gnss = read_gnss(args.gnss)
    optimized = optimize_pose_graph(
        trajectory.names,
        trajectory.poses_wc,
        gnss,
        gnss_sigma=args.gnss_sigma,
        odom_sigma_rot=args.odom_sigma_rot,
        odom_sigma_trans=args.odom_sigma_trans,
    )
    write_trajectory(args.work_dir, trajectory.names, trajectory.ids, optimized, suffix="_optimized")
    print(f"Optimized trajectory saved to {args.work_dir / 'trajectory_optimized.txt'}")

    if viewer is not None:
        viewer.show_optimized(
            np.array([T[:3, 3] for T in optimized]),
            np.array([gnss[n] for n in trajectory.names if n in gnss]).reshape(-1, 3),
        )


def main() -> None:
    args = parse_args()

    main_script = COLMAP_ODOMETRY / "main.py"
    if not main_script.exists():
        sys.exit(
            f"colmap-odometry not found in {COLMAP_ODOMETRY}.\n"
            "Run: git submodule update --init --recursive"
        )

    for name in ("images", "config", "camera", "rig", "gnss"):
        path = getattr(args, name)
        if path is not None and not path.exists():
            sys.exit(f"--{name} not found: {path}")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # colmap-odometry imports its modules as `src.odometry...`
    sys.path.insert(0, str(COLMAP_ODOMETRY))

    trajectory = OdometryTrajectory()
    viewer = TrajectoryViewer() if args.gui else None
    attach(trajectory, viewer)

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

    # Rewrite upstream trajectory.txt / images.txt with header and orientation
    write_trajectory(args.work_dir, trajectory.names, trajectory.ids, trajectory.poses_wc,
                     covariances=trajectory.covariances)

    if args.gnss is not None:
        optimize_with_gnss(args, trajectory, viewer)

    if viewer is not None:
        viewer.keep_open()


if __name__ == "__main__":
    main()
