# Rotations and translations in world reference system. Rotations in euler angles (degrees) and translations in meters.

import argparse
import os
import math

import pyquaternion

try:
    import yaml
except Exception:
    yaml = None

cameras = {
    "cam0": {
        "rotation": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0]
    },
    "cam1": {
        "rotation": [0.0, 0.0, 0.0],
        "translation": [0.12, 0.0, 0.0]
    },
    "cam2": {
        "rotation": [0.0, 90.0, 0.0],
        "translation": [0.0, 0.0, 0.0]
    },
    "cam3": {
        "rotation": [0.0, 90.0, 0.0],
        "translation": [0.12, 0.0, 0.0]
    }
}


def euler_deg_to_quaternion(euler_deg):
    """Convert Euler angles in degrees (roll, pitch, yaw) to a pyquaternion.Quaternion.

    Assumes angles are in degrees and the rotation order is roll (X), pitch (Y), yaw (Z).
    """
    roll, pitch, yaw = [math.radians(float(a)) for a in euler_deg]

    # Build quaternions for each axis rotation and combine them.
    # We apply roll (X) first, then pitch (Y), then yaw (Z), so overall q = qz * qy * qx.
    qx = pyquaternion.Quaternion(axis=[1, 0, 0], angle=roll)
    qy = pyquaternion.Quaternion(axis=[0, 1, 0], angle=pitch)
    qz = pyquaternion.Quaternion(axis=[0, 0, 1], angle=yaw)

    return qz * qy * qx


def build_rig_dict(cameras_dict):
    rig = {}
    for name, cam in cameras_dict.items():
        rot = cam.get("rotation", [0.0, 0.0, 0.0])
        trans = cam.get("translation", [0.0, 0.0, 0.0])
        q = euler_deg_to_quaternion(rot)
        # store quaternion as [x, y, z, w] to match requested output format
        trans = -q.inverse.rotation_matrix @ trans
        q = q.inverse
        rig[name] = {
            "translation": [float(x) for x in trans],
            "rotation": [float(q.x), float(q.y), float(q.z), float(q.w)]
        }
    return rig


def _fmt_num(v, precision=4):
    # format number with up to `precision` decimals, but keep at least one decimal place
    s = f"{v:.{precision}f}"
    # strip unnecessary trailing zeros and optional trailing dot
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    # ensure at least one decimal place
    if '.' not in s:
        s = s + '.0'
    return s


def write_yaml(rig_dict, output_path):
    # write in the exact requested style
    lines = []
    for i, (name, cam) in enumerate(rig_dict.items()):
        lines.append(f"{name}:")
        trans_str = ', '.join(_fmt_num(x, precision=4) for x in cam['translation'])
        rot_str = ', '.join(_fmt_num(x, precision=4) for x in cam['rotation'])
        lines.append(f"  rotation: [{rot_str}]")
        lines.append(f"  translation: [{trans_str}]")
        # blank line between cameras
        if i != len(rig_dict) - 1:
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare rig YAML with translations and quaternions from Euler angles"
    )
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("-f", "--filename", default="rig.yaml", help="Output filename")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, args.filename)

    rig = build_rig_dict(cameras)
    write_yaml(rig, out_path)

    print(f"Wrote rig YAML to: {out_path}")


if __name__ == "__main__":
    main()