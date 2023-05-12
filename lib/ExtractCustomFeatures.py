import os
from pathlib import Path
import numpy as np
from lib import db_colmap
import cv2


def ALIKE(img_name, desc_folder):
    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path

    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]

    return kp, desc, kp_numb


def KeyNet(img_name, desc_folder):
    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path

    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]

    return kp, desc, kp_numb


def SuperPoint(img_name, desc_folder):
    img_path = desc_folder / img_name
    kp = np.load(os.path.join("{}.kps.npy".format(img_path)))
    desc = np.load("{}.des.npy".format(img_path))
    kp_numb = kp.shape[0]

    return kp, desc, kp_numb


def ExtractCustomFeatures(
    CUSTOM_DETECTOR, PATH_TO_LOCAL_FEATURES, DATABASE, KEYFRAMES_DIR, keyframes_list
):
    # inverted_img_dict = {v: k for k, v in img_dict.items()}
    kfrms = os.listdir(KEYFRAMES_DIR)
    kfrms.sort()
    db = db_colmap.COLMAPDatabase.connect(str(DATABASE))
    db.create_tables()

    existing_images = dict(
        (image_id, name)
        for image_id, name in db.execute("SELECT image_id, name FROM images")
    )

    existing_images = list(existing_images.values())

    # Create cameras.
    model1, width1, height1, params1 = (
        4,
        752,
        480,
        np.array(
            (
                458.654,
                457.296,
                367.215,
                248.375,
                -0.28340811,
                0.07395907,
                0.00019359,
                1.76187114e-05,
            )
        ),
    )
    camera_id1 = db.add_camera(model1, width1, height1, params1)

    if CUSTOM_DETECTOR == "ALIKE":
        for img in kfrms:
            if img not in existing_images:
                kp, desc, kp_numb = ALIKE(
                    inverted_img_dict[img][:-3] + "png", PATH_TO_LOCAL_FEATURES
                )
                kp = kp[:, 0:2]

                zero_matrix = np.zeros((np.shape(desc)[0], 64))
                desc = np.append(desc, zero_matrix, axis=1)
                desc.astype(np.float32)
                desc = np.absolute(desc)

                desc = desc * 512 / np.linalg.norm(desc, axis=1).reshape((-1, 1))
                desc = np.round(desc)
                desc = np.array(desc, dtype=np.uint8)

                one_matrix = np.ones((np.shape(kp)[0], 1))
                kp = np.append(kp, one_matrix, axis=1)
                zero_matrix = np.zeros((np.shape(kp)[0], 1))
                kp = np.append(kp, zero_matrix, axis=1).astype(np.float32)

                img_id = db.add_image(img, camera_id1)
                db.add_keypoints(img_id, kp)
                db.add_descriptors(img_id, desc)
                db.commit()

    elif CUSTOM_DETECTOR == "ORB":
        for img in kfrms:
            if img not in existing_images:
                im = cv2.imread(str(KEYFRAMES_DIR / img), cv2.IMREAD_GRAYSCALE)
                orb = cv2.ORB_create(nfeatures=1024)
                kp = orb.detect(im, None)
                kp, des = orb.compute(im, kp)
                all_kpts = np.zeros((len(kp), 2))
                for j in range(len(kp)):
                    all_kpts[j, 0], all_kpts[j, 1] = (
                        cv2.KeyPoint_convert(kp)[j][0],
                        cv2.KeyPoint_convert(kp)[j][1],
                    )
                one_matrix = np.ones((len(kp), 1))
                all_kpts = np.append(all_kpts, one_matrix, axis=1)
                zero_matrix = np.zeros((len(kp), 1))
                all_kpts = np.append(all_kpts, zero_matrix, axis=1).astype(np.float32)

                zero_matrix = np.zeros((np.shape(des)[0], 96))
                des = np.append(des, zero_matrix, axis=1)
                des.astype(np.float32)
                des = np.absolute(des)

                des = des * 512 / np.linalg.norm(des, axis=1).reshape((-1, 1))
                des = np.round(des)
                des = np.array(des, dtype=np.uint8)

                img_id = db.add_image(img, camera_id1)
                db.add_keypoints(img_id, all_kpts)
                db.add_descriptors(img_id, des)
                db.commit()

    elif CUSTOM_DETECTOR == "KEYNET":
        for img in kfrms:
            if img not in existing_images:
                kp, desc, kp_numb = KeyNet(
                    inverted_img_dict[img][:-3] + "png", PATH_TO_LOCAL_FEATURES
                )
                kp = kp[:, 0:2]

                desc.astype(np.float32)
                desc = np.absolute(desc)

                desc = desc * 512 / np.linalg.norm(desc, axis=1).reshape((-1, 1))
                desc = np.round(desc)
                desc = np.array(desc, dtype=np.uint8)

                one_matrix = np.ones((np.shape(kp)[0], 1))
                kp = np.append(kp, one_matrix, axis=1)
                zero_matrix = np.zeros((np.shape(kp)[0], 1))
                kp = np.append(kp, zero_matrix, axis=1).astype(np.float32)

                img_id = db.add_image(img, camera_id1)
                db.add_keypoints(img_id, kp)
                db.add_descriptors(img_id, desc)
                db.commit()

    elif CUSTOM_DETECTOR == "SUPERPOINT":
        for img in kfrms:
            print("existing_images", existing_images)
            print("img", img)
            if img not in existing_images:
                keyframe_obj = keyframes_list.get_keyframe_by_name(img)
                # keyframe_obj = list(filter(lambda obj: obj.keyframe_name == img, keyframes_list))[0]
                image_name = keyframe_obj.image_name
                kp, desc, kp_numb = SuperPoint(image_name, PATH_TO_LOCAL_FEATURES)
                kp = kp[:, 0:2]
                desc = desc[:, :128]

                desc.astype(np.float32)
                desc = np.absolute(desc)

                desc = desc * 512 / np.linalg.norm(desc, axis=1).reshape((-1, 1))
                desc = np.round(desc)
                desc = np.array(desc, dtype=np.uint8)

                one_matrix = np.ones((np.shape(kp)[0], 1))
                kp = np.append(kp, one_matrix, axis=1)
                zero_matrix = np.zeros((np.shape(kp)[0], 1))
                kp = np.append(kp, zero_matrix, axis=1).astype(np.float32)

                img_id = db.add_image(img, camera_id1)
                db.add_keypoints(img_id, kp)
                db.add_descriptors(img_id, desc)
                db.commit()

    else:
        print("Local features not supported. Exit.")
        quit()

    db.close()
