import numpy as np
import pyquaternion

input_images = r"/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/input/images.txt"
output_images = r"/media/threedom/Seagate Expansion Drive/3DOM/MMT25_Stelvio/data/superpoint_results_calibration_colmap_full_self/working/triangulation/output/images.txt"

transf_matrix = np.array([
    [0.9995029602084016, 0.028634965864050926, -0.013186025352539845, -0.2592356043125846],
    [-0.028563741162251, 0.9995765177535869, 0.005558583130609153, 0.00962644749119568],
    [0.013339611143099835, -0.005179178078479528, 0.9998976102026512, 0.007099331097474101],
    [0.0, 0.0, 0.0, 1.0]
])
transf_matrix = np.linalg.inv(transf_matrix)
#transf_matrix = np.eye(4)
#transf_matrix[:3, 3] = np.array([0.50, 0.0, 0.0])

k = 0
with open(input_images, 'r') as f, open(output_images, 'w') as f_out:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        try:
            id, qw, qx, qy, qz, tx, ty, tz, camid, image = line.split(" ", 9)
        except:
            continue
        #f_out.write(f"{id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camid} cam0/{image}\n\n")
        q = pyquaternion.Quaternion([float(qw), float(qx), float(qy), float(qz)])
        t = np.array([float(tx), float(ty), float(tz)])
        R = q.rotation_matrix
        R_inv = np.linalg.inv(R)
        C = -R_inv @ t
        #T = np.eye(4)
        #T[:3, :3] = R
        #T[:3, 3] = C
        #T_cam2 = T @ transf_matrix
        #R_cam2 = T_cam2[:3, :3]
        #C_cam2 = T_cam2[:3, 3]
        #t_cam2 = -R_cam2 @ C_cam2
        #qw_cam2, qx_cam2, qy_cam2, qz_cam2 = pyquaternion.Quaternion(matrix=R_cam2).elements

        C_cam2 = C + R_inv @ transf_matrix[:3, 3]
        R_cam2 = transf_matrix[:3, :3] @ R
        qw_cam2, qx_cam2, qy_cam2, qz_cam2 = pyquaternion.Quaternion(matrix=R_cam2).elements
        t_cam2 = -R_cam2 @ C_cam2

        #f_out.write(f"{C[0]} {C[1]} {C[2]} 0 255 255\n")
        #f_out.write(f"{C_cam2[0]} {C_cam2[1]} {C_cam2[2]} 255 0 0\n")

        k += 1
        f_out.write(f"{k} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 cam0/{image}\n\n")
        k += 1
        f_out.write(f"{k} {qw_cam2} {qx_cam2} {qy_cam2} {qz_cam2} {t_cam2[0]} {t_cam2[1]} {t_cam2[2]} 2 cam1/{image}\n\n")


