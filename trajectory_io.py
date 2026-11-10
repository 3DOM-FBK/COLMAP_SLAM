"""Write trajectories to text files."""

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

TRAJECTORY_HEADER = (
    "# image_name tx ty tz qw qx qy qz{extra}\n"
    "# camera-to-world pose: (tx, ty, tz) camera center, (qw, qx, qy, qz) camera orientation in world\n"
)


def write_trajectory(work_dir: Path, names: list, ids: list, poses_wc: list,
                     suffix: str = "", covariances: list = None) -> None:
    """Write trajectory{suffix}.txt (with header) and images{suffix}.txt (COLMAP format).

    Args:
        work_dir: output folder.
        names, ids: image name and COLMAP image id of each pose.
        poses_wc: 4x4 camera-to-world matrices.
        suffix: appended to the file names (e.g. "_optimized").
        covariances: optional per-pose value written as last column `cov`.
    """
    extra = " cov" if covariances is not None else ""
    traj_lines = [TRAJECTORY_HEADER.format(extra=extra)]
    img_lines = []
    for k, (name, image_id, T) in enumerate(zip(names, ids, poses_wc)):
        c = T[:3, 3]
        q_wc = Quaternion(matrix=T[:3, :3], atol=1e-6)
        line = f"{name} {c[0]} {c[1]} {c[2]} {q_wc[0]} {q_wc[1]} {q_wc[2]} {q_wc[3]}"
        if covariances is not None:
            line += f" {covariances[k]}"
        traj_lines.append(line + "\n")

        q_cw = q_wc.inverse
        t_cw = -q_cw.rotation_matrix @ c
        img_lines.append(
            f"{image_id} {q_cw[0]} {q_cw[1]} {q_cw[2]} {q_cw[3]} "
            f"{t_cw[0]} {t_cw[1]} {t_cw[2]} 1 {name}\n\n"
        )
    (work_dir / f"trajectory{suffix}.txt").write_text("".join(traj_lines))
    (work_dir / f"images{suffix}.txt").write_text("".join(img_lines))
