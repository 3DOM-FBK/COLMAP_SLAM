
import numpy as np
import pyquaternion


rot_cam0 = [
    0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
    0.0, 0.0, 0.0, 1.0,
]

rot_cam1 = [
    0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
    0.0, 0.0, 0.0, 1.0,
]


def as_transform(flat_values):
    return np.asarray(flat_values, dtype=np.float64).reshape(4, 4)


def relative_pose(cam0, cam1):
    T_cam0 = as_transform(cam0)
    T_cam1 = as_transform(cam1)
    T_rel = np.linalg.inv(T_cam0) @ T_cam1
    return T_rel


def main():
    T_rel = relative_pose(rot_cam0, rot_cam1)

    print("Relative pose of cam1 w.r.t cam0:")
    print(np.array2string(T_rel, precision=15, suppress_small=False))

    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    q_rel = pyquaternion.Quaternion(matrix=R_rel)

    print("\nR_rel =")
    print(np.array2string(R_rel, precision=15, suppress_small=False))
    print("\nq_rel = [x, y, z, w]")
    print([q_rel.x, q_rel.y, q_rel.z, q_rel.w])
    print("\nt_rel =")
    print(np.array2string(t_rel, precision=15, suppress_small=False))


if __name__ == "__main__":
    main()