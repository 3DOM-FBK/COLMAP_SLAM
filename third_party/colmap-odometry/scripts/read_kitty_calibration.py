#!/usr/bin/env python3
"""
Read KITTI calibration file and compute transformation between cameras.

KITTI calibration format:
P0: fx 0 cx tx 0 fy cy ty 0 0 1 tz (left grayscale camera)
P1: fx 0 cx tx 0 fy cy ty 0 0 1 tz (right grayscale camera)  
P2: fx 0 cx tx 0 fy cy ty 0 0 1 tz (left color camera)
P3: fx 0 cx tx 0 fy cy ty 0 0 1 tz (right color camera)

The projection matrix P = K * [R|t] where:
- K is the intrinsic matrix
- R is the rotation matrix 
- t is the translation vector

For stereo pairs, we can extract the baseline and compute transformations.
"""

import argparse
import numpy as np
import os
import sys


def read_kitti_calib(calib_file):
    """
    Read KITTI calibration file and parse projection matrices.
    
    Args:
        calib_file (str): Path to KITTI calibration file
        
    Returns:
        dict: Dictionary containing projection matrices P0, P1, P2, P3
    """
    
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file '{calib_file}' does not exist.")
        sys.exit(1)
    
    calib_data = {}
    
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                key, values = line.split(':', 1)
                key = key.strip()
                # Parse the 12 values and reshape to 3x4 matrix
                values = [float(x) for x in values.strip().split()]
                if len(values) == 12:
                    calib_data[key] = np.array(values).reshape(3, 4)
                else:
                    print(f"Warning: Expected 12 values for {key}, got {len(values)}")
    
    return calib_data


def extract_camera_params(P):
    """
    Extract camera parameters from projection matrix P = K * [R|t].
    
    Args:
        P (np.array): 3x4 projection matrix
        
    Returns:
        tuple: (K, R, t) - intrinsic matrix, rotation matrix, translation vector
    """
    
    # For KITTI, the cameras are rectified, so R = I for the reference camera
    # P = K * [I|t] for rectified cameras
    
    # Extract intrinsic matrix K (upper 3x3)
    K = P[:, :3]
    
    # For rectified cameras, we can assume R = I
    # Translation can be computed as t = K^(-1) * P[:, 3]
    t = np.linalg.inv(K) @ P[:, 3]
    
    return K, np.eye(3), t


def compute_transformation_2_from_3(P2, P3):
    """
    Compute transformation from camera 3 to camera 2.
    
    Args:
        P2 (np.array): Projection matrix of camera 2
        P3 (np.array): Projection matrix of camera 3
        
    Returns:
        tuple: (R, t) - rotation and translation from camera 3 to camera 2
    """
    
    # Extract camera parameters
    K2, R2, t2 = extract_camera_params(P2)
    K3, R3, t3 = extract_camera_params(P3)
    
    print("Camera 2 parameters:")
    print(f"K2 =\n{K2}")
    print(f"t2 = {t2}")
    print()
    
    print("Camera 3 parameters:")
    print(f"K3 =\n{K3}")
    print(f"t3 = {t3}")
    print()
    
    # For rectified stereo cameras, R2 = R3 = I
    # The transformation from camera 3 to camera 2 is:
    # R_2_from_3 = R2 * R3^T = I * I^T = I
    # t_2_from_3 = t2 - R_2_from_3 * t3 = t2 - t3
    
    R_2_from_3 = R2 @ R3.T  # Should be identity for rectified cameras
    t_2_from_3 = t2 - R_2_from_3 @ t3
    
    return R_2_from_3, t_2_from_3


def compute_baseline_and_disparity(P2, P3):
    """
    Compute baseline between cameras and disparity-to-depth conversion factor.
    
    Args:
        P2 (np.array): Projection matrix of camera 2 (left)
        P3 (np.array): Projection matrix of camera 3 (right)
        
    Returns:
        tuple: (baseline, focal_length) for stereo computation
    """
    
    K2, _, t2 = extract_camera_params(P2)
    K3, _, t3 = extract_camera_params(P3)
    
    # Baseline is the difference in x translation (assuming horizontal stereo)
    baseline = abs(t2[0] - t3[0])
    
    # Focal length from intrinsic matrix
    focal_length = K2[0, 0]  # fx
    
    return baseline, focal_length


def main():
    parser = argparse.ArgumentParser(
        description="Read KITTI calibration file and compute camera transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python read_kitty_calibration.py calib.txt
  python read_kitty_calibration.py /path/to/kitti/calib.txt

KITTI Calibration Format:
  P0: fx 0 cx tx 0 fy cy ty 0 0 1 tz  (left grayscale)
  P1: fx 0 cx tx 0 fy cy ty 0 0 1 tz  (right grayscale)
  P2: fx 0 cx tx 0 fy cy ty 0 0 1 tz  (left color)
  P3: fx 0 cx tx 0 fy cy ty 0 0 1 tz  (right color)
        """
    )
    
    parser.add_argument('calib_file', 
                       help='Path to KITTI calibration file')
    
    args = parser.parse_args()
    
    # Read calibration data
    print(f"Reading KITTI calibration from: {args.calib_file}")
    calib_data = read_kitti_calib(args.calib_file)
    
    # Print all projection matrices
    for key in ['P0', 'P1', 'P2', 'P3']:
        if key in calib_data:
            print(f"\n{key} =")
            print(calib_data[key])
    
    # Check if P2 and P3 exist (color cameras)
    if 'P2' not in calib_data or 'P3' not in calib_data:
        print("\nError: P2 and P3 matrices are required to compute transformation.")
        sys.exit(1)
    
    # Compute transformation from camera 3 to camera 2
    print("\n" + "="*50)
    print("COMPUTING TRANSFORMATION 2_FROM_3")
    print("="*50)
    
    R_2_from_3, t_2_from_3 = compute_transformation_2_from_3(calib_data['P2'], calib_data['P3'])
    
    print("Transformation from camera 3 to camera 2:")
    print(f"R_2_from_3 =\n{R_2_from_3}")
    print(f"t_2_from_3 = {t_2_from_3}")
    print()
    
    # Compute stereo baseline
    baseline, focal_length = compute_baseline_and_disparity(calib_data['P2'], calib_data['P3'])
    print(f"Stereo baseline: {baseline:.6f} meters")
    print(f"Focal length: {focal_length:.2f} pixels")
    print(f"Disparity-to-depth factor (f*b): {focal_length * baseline:.6f}")
    
    # Create transformation matrix
    T_2_from_3 = np.eye(4)
    T_2_from_3[:3, :3] = R_2_from_3
    T_2_from_3[:3, 3] = t_2_from_3
    
    print(f"\n4x4 Transformation matrix T_2_from_3:")
    print(T_2_from_3)


if __name__ == "__main__":
    main()
