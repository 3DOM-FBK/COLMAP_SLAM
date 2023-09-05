import numpy as np
from pyquaternion import quaternion


def ExportCameras(external_cameras_path, keyframes_dict):
    lines = []
    lines.append("IMAGE_ID X Y Z NX NY NZ FOCAL_LENGTH EULER_ROTATION_MATRIX\n")
    camera_dict = {}
    k = 0

    with open(external_cameras_path, "r") as file:
        for line in file:
            k = k + 1
            line = line.strip()
            first_elem, waste = line.split(" ", 1)
            if first_elem == "#":
                pass

            elif k % 2 != 0:
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.split(
                    " ", 9
                )

                cam_number, keyframe = name.split("/", 1)

                
                q = np.array([float(qw), float(qx), float(qy), float(qz)])
                t = np.array([[float(tx)], [float(ty)], [float(tz)]])
                q_matrix = quaternion.Quaternion(q).transformation_matrix
                q_matrix = q_matrix[0:3, 0:3]
                camera_location = np.dot(-q_matrix.transpose(), t)
                camera_direction = np.dot(
                    q_matrix.transpose(), np.array([[0], [0], [1]])
                )
                lines.append(
                    "{} {} {} {} {} {} {} {} 50 {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                        name,
                        "imgs/" + cam_number + "/" + str(keyframes_dict[keyframe]['image_name']),
                        camera_location[0, 0],
                        camera_location[1, 0],
                        camera_location[2, 0],
                        camera_direction[0, 0],
                        camera_direction[1, 0],
                        camera_direction[2, 0],
                        q_matrix[0, 0],
                        q_matrix[0, 1],
                        q_matrix[0, 2],
                        "0",
                        q_matrix[1, 0],
                        q_matrix[1, 1],
                        q_matrix[1, 2],
                        "0",
                        q_matrix[2, 0],
                        q_matrix[2, 1],
                        q_matrix[2, 2],
                        "0",
                        "0",
                        "0",
                        "0",
                        "1",
                    )
                )
                #id_camera = int(name[5:-4])
                camera_dict[name] = (
                    (
                        camera_location[0, 0],
                        camera_location[1, 0],
                        camera_location[2, 0],
                    ),
                    (q, t),
                )
    
    return lines, camera_dict
