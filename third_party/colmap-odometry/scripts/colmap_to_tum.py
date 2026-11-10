#!/usr/bin/env python3
"""
Convert COLMAP images.txt file to TUM trajectory format.

COLMAP images.txt format:
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# POINTS2D[] as (X, Y, POINT3D_ID)

TUM trajectory format:
timestamp tx ty tz qx qy qz qw

Note: COLMAP poses are written on alternating lines (image data, then 2D points).
The camera center is computed as C = -R^(-1) * t where R is the rotation matrix from quaternion.
"""

import argparse
import numpy as np
from pyquaternion import Quaternion
import os
import sys


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix using pyquaternion."""
    quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    return quat.rotation_matrix


def colmap_to_tum(images_file, output_file, timestamps_file=None):
    """
    Convert COLMAP images.txt to TUM format.
    
    Args:
        images_file (str): Path to COLMAP images.txt file
        output_file (str): Path to output TUM trajectory file
        timestamps_file (str, optional): Path to timestamps file (one timestamp per line)
    """
    
    if not os.path.exists(images_file):
        print(f"Error: Images file '{images_file}' does not exist.")
        sys.exit(1)
    
    # Load timestamps from file if provided
    external_timestamps = []
    if timestamps_file:
        if not os.path.exists(timestamps_file):
            print(f"Error: Timestamps file '{timestamps_file}' does not exist.")
            sys.exit(1)
        
        with open(timestamps_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        external_timestamps.append(float(line))
                    except ValueError:
                        print(f"Warning: Could not parse timestamp: {line}")
        
        print(f"Loaded {len(external_timestamps)} timestamps from {timestamps_file}")
        external_timestamps = external_timestamps[1:]

    tum_poses = []
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    # Parse COLMAP images.txt (skip comments and empty lines)
    i = 0
    pose_index = 0  # Index for external timestamps
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if line.startswith('#') or not line:
            i += 1
            continue
        
        # Parse image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) < 9:  # Minimum required: name + 4 quaternion + 3 translation + camera_id
            print(f"Warning: Skipping malformed line {i+1}: {line}")
            i += 1
            continue
        
        try:
            # Try to parse as standard COLMAP format first
            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]
            except ValueError:
                # If image_id parsing fails, the format might be different
                # Sometimes COLMAP outputs image name first: NAME, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_ID
                if parts[0].endswith('.png') or parts[0].endswith('.jpg'):
                    image_name = parts[0]
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8]) if len(parts) > 8 else 1
                    # Try to get image_id from end, or extract from filename
                    try:
                        image_id = int(parts[9]) if len(parts) > 9 else None
                    except (ValueError, IndexError):
                        image_id = None
                    
                    # If no valid image_id, try to extract from filename
                    if image_id is None:
                        try:
                            name_base = image_name.split('.png')[0].split('.jpg')[0]
                            if name_base.isdigit():
                                image_id = int(name_base)
                            else:
                                image_id = 1  # Default fallback
                        except:
                            image_id = 1  # Default fallback
                else:
                    raise ValueError(f"Cannot parse line format: {line}")
            
            # Use external timestamps if provided, otherwise extract from image name
            if external_timestamps:
                if pose_index < len(external_timestamps):
                    timestamp = external_timestamps[pose_index]
                    pose_index += 1
                else:
                    print(f"Warning: No more timestamps available for image {image_id}. Using image_id.")
                    timestamp = float(image_id)
            else:
                # Extract timestamp from image name if it contains timestamp
                # Format like: 01744280373_691761000.png -> extract the timestamp parts
                # Or format like: 000594.png -> use as simple number
                if '_' in image_name and (image_name.endswith('.png') or image_name.endswith('.jpg')):
                    try:
                        # Remove file extension and split by underscore
                        name_base = image_name.split('.png')[0].split('.jpg')[0]
                        name_parts = name_base.split('_')
                        if len(name_parts) == 2:
                            # Convert to seconds: first part + second part as nanoseconds
                            seconds = int(name_parts[0])
                            nanoseconds = int(name_parts[1])
                            timestamp = float(seconds) + float(nanoseconds) / 1e9
                        else:
                            timestamp = float(image_id)
                    except (ValueError, IndexError):
                        timestamp = float(image_id)
                elif (image_name.endswith('.png') or image_name.endswith('.jpg')):
                    try:
                        # Try to extract number from filename like 000594.png
                        name_base = image_name.split('.png')[0].split('.jpg')[0]
                        if name_base.isdigit():
                            timestamp = float(name_base)
                        else:
                            timestamp = float(image_id)
                    except (ValueError, IndexError):
                        timestamp = float(image_id)
                else:
                    timestamp = float(image_id)
            
            # COLMAP quaternion format is (w, x, y, z)
            # COLMAP translation is the camera position in world coordinates (t = -R * C)
            # We need to compute camera center: C = -R^(-1) * t
            
            # Create quaternion and get rotation matrix
            quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
            R = quat.rotation_matrix
            
            # Compute camera center: C = -R^(-1) * t = -R^T * t
            t_colmap = np.array([tx, ty, tz])
            camera_center = -R.T @ t_colmap
            
            # For TUM format, we need the quaternion in (x, y, z, w) format
            # and we use the camera center as translation
            tum_poses.append({
                'timestamp': timestamp,  # Use extracted timestamp
                'tx': camera_center[0],
                'ty': camera_center[1], 
                'tz': camera_center[2],
                'qx': -qx,
                'qy': -qy,
                'qz': -qz,
                'qw': qw
            })
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing line {i+1}: {e}")
        
        # Skip next line (2D points data)
        i += 2
    
    # Sort by timestamp and write to output file
    tum_poses.sort(key=lambda x: x['timestamp'])
    
    with open(output_file, 'w') as f:
        f.write("# TUM trajectory format\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for pose in tum_poses:
            # Use standard float formatting for all timestamps
            f.write(f"{pose['timestamp']:.6f} {pose['tx']:.6f} {pose['ty']:.6f} {pose['tz']:.6f} "
                   f"{pose['qx']:.6f} {pose['qy']:.6f} {pose['qz']:.6f} {pose['qw']:.6f}\n")
    
    print(f"Converted {len(tum_poses)} poses from '{images_file}' to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP images.txt file to TUM trajectory format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python colmap_to_tum.py images.txt trajectory.txt
  python colmap_to_tum.py images.txt trajectory.txt --timestamps timestamps.txt
  python colmap_to_tum.py /path/to/colmap/images.txt /path/to/output/tum_trajectory.txt -t /path/to/timestamps.txt

COLMAP format (images.txt):
  IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
  (followed by 2D points on next line)

TUM format:
  timestamp tx ty tz qx qy qz qw

Timestamps file format (optional):
  0.000000e+00
  1.038725e-01
  2.077453e-01
  ...
        """
    )
    
    parser.add_argument('images_file', 
                       help='Path to COLMAP images.txt file')
    
    parser.add_argument('output_file',
                       help='Path to output TUM trajectory file')
    
    parser.add_argument('--timestamps', '-t',
                       help='Path to timestamps file (one timestamp per line in scientific notation)',
                       default=None)
    
    args = parser.parse_args()
    
    # Convert COLMAP to TUM
    colmap_to_tum(args.images_file, args.output_file, args.timestamps)


if __name__ == "__main__":
    main()
