"""Pose-graph optimization of the colmap-odometry trajectory with GNSS priors (gtsam).

Poses are camera-to-world transforms T_wc (rotation R_wc, camera center t).
The graph contains:
  * one BetweenFactorPose3 per consecutive pair of localized frames (odometry),
  * one GPSFactor per frame that has a GNSS position,
  * a weak prior on the first pose, only to fix the rotation gauge when the
    GNSS positions are (nearly) collinear.
"""

from pathlib import Path

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X


def read_gnss(path: Path) -> dict:
    """Read a GNSS file with lines `image_name x y z [...]` (extra columns are ignored)."""
    gnss = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#"):
            continue
        gnss[parts[0]] = np.array([float(v) for v in parts[1:4]])
    return gnss


def _align_rigid(src: np.ndarray, dst: np.ndarray):
    """Rotation R and translation t that best map src points onto dst (Kabsch)."""
    if len(src) < 3:
        return np.eye(3), (dst - src).mean(axis=0)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def optimize_pose_graph(
    names: list,
    poses_wc: list,
    gnss: dict,
    gnss_sigma: float = 0.05,
    odom_sigma_rot: float = 0.01,
    odom_sigma_trans: float = 0.05,
) -> list:
    """Optimize the odometry poses with GNSS position priors.

    Args:
        names: image name of each pose.
        poses_wc: 4x4 camera-to-world matrices from the odometry.
        gnss: image name -> GNSS position (same world frame expected; if not,
            the odometry is rigidly aligned to it for the initial guess).
        gnss_sigma: GNSS position standard deviation [m].
        odom_sigma_rot: relative-pose rotation standard deviation [rad].
        odom_sigma_trans: relative-pose translation standard deviation [m].

    Returns:
        Optimized 4x4 camera-to-world matrices, one per input pose.
    """
    poses = [gtsam.Pose3(T) for T in poses_wc]
    matched = [i for i, n in enumerate(names) if n in gnss]
    if not matched:
        raise ValueError("No GNSS position matches an image name of the trajectory")

    # Initial guess: odometry rigidly aligned to the GNSS frame
    R, t = _align_rigid(
        np.array([poses[i].translation() for i in matched]),
        np.array([gnss[names[i]] for i in matched]),
    )
    align = gtsam.Pose3(gtsam.Rot3(R), t)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([odom_sigma_rot] * 3 + [odom_sigma_trans] * 3)
    )
    gnss_noise = gtsam.noiseModel.Isotropic.Sigma(3, gnss_sigma)
    gauge_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0] * 3 + [10.0] * 3))

    for i, pose in enumerate(poses):
        initial.insert(X(i), align.compose(pose))
        if i > 0:
            graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), poses[i - 1].between(pose), odom_noise))
        if names[i] in gnss:
            graph.add(gtsam.GPSFactor(X(i), gtsam.Point3(gnss[names[i]]), gnss_noise))
    graph.add(gtsam.PriorFactorPose3(X(0), initial.atPose3(X(0)), gauge_noise))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam.LevenbergMarquardtParams()).optimize()

    print(
        f"Pose graph: {len(poses)} poses, {len(poses) - 1} odometry factors, "
        f"{len(matched)} GNSS factors | error {graph.error(initial):.4f} -> {graph.error(result):.4f}"
    )
    return [result.atPose3(X(i)).matrix() for i in range(len(poses))]


def write_trajectory(work_dir: Path, names: list, ids: list, poses_wc: list, suffix: str = "_optimized") -> None:
    """Write trajectory{suffix}.txt (image x y z) and images{suffix}.txt (COLMAP format)."""
    traj_lines, img_lines = [], []
    for name, image_id, T in zip(names, ids, poses_wc):
        c = T[:3, 3]
        R_cw = T[:3, :3].T
        q = gtsam.Rot3(R_cw).toQuaternion()
        t_cw = -R_cw @ c
        traj_lines.append(f"{name} {c[0]} {c[1]} {c[2]}\n")
        img_lines.append(
            f"{image_id} {q.w()} {q.x()} {q.y()} {q.z()} {t_cw[0]} {t_cw[1]} {t_cw[2]} 1 {name}\n\n"
        )
    (work_dir / f"trajectory{suffix}.txt").write_text("".join(traj_lines))
    (work_dir / f"images{suffix}.txt").write_text("".join(img_lines))
