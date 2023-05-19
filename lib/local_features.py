import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from easydict import EasyDict as edict

from lib import db_colmap
from lib.thirdparty.alike.alike import ALike, configs


class LocalFeatures:
    def __init__(
        self,
        method: str,
        n_features: int,
        cfg: dict = None,
    ) -> None:
        self.n_features = n_features
        self.method = method

        self.kpts = {}
        self.descriptors = {}

        # If method is ALIKE, load Alike model weights
        if self.method == "ALIKE":
            self.alike_cfg = cfg
            self.model = ALike(
                **configs[self.alike_cfg.model],
                device=self.alike_cfg.device,
                top_k=self.alike_cfg.top_k,
                scores_th=self.alike_cfg.scores_th,
                n_limit=self.alike_cfg.n_limit,
            )

        elif self.method == "ORB":
            self.orb_cfg = cfg

        elif self.method == "KeyNetAffNetHardNet":
            self.kornia_cfg = cfg
            self.device = torch.device("cuda")  # TODO set to cuda

    def ORB(self, images: List[Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        for im_path in images:
            im_path = Path(im_path)
            im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create(
                nfeatures=self.n_features,
                scaleFactor=self.orb_cfg.scaleFactor,
                nlevels=self.orb_cfg.nlevels,
                edgeThreshold=self.orb_cfg.edgeThreshold,
                firstLevel=self.orb_cfg.firstLevel,
                WTA_K=self.orb_cfg.WTA_K,
                scoreType=self.orb_cfg.scoreType,
                patchSize=self.orb_cfg.patchSize,
                fastThreshold=self.orb_cfg.fastThreshold,
            )
            kp = orb.detect(im, None)
            kp, des = orb.compute(im, kp)
            kpts = cv2.KeyPoint_convert(kp)

            one_matrix = np.ones((len(kp), 1))
            kpts = np.append(kpts, one_matrix, axis=1)
            zero_matrix = np.zeros((len(kp), 1))
            kpts = np.append(kpts, zero_matrix, axis=1).astype(np.float32)

            zero_matrix = np.zeros((des.shape[0], 96))
            des = np.append(des, zero_matrix, axis=1).astype(np.float32)
            des = np.absolute(des)
            des = des * 512 / np.linalg.norm(des, axis=1).reshape((-1, 1))
            des = np.round(des)
            des = np.array(des, dtype=np.uint8)

            self.kpts[im_path.stem] = kpts
            self.descriptors[im_path.stem] = des

        return self.kpts, self.descriptors

    def ALIKE(self, images: List[Path]):
        for im_path in images:
            img = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB)
            features = self.model(img, sub_pixel=self.alike_cfg.subpixel)

            self.kpts[im_path.stem] = features["keypoints"]
            self.descriptors[im_path.stem] = features["descriptors"]

        return self.kpts, self.descriptors

    def load_torch_image(self, fname):
        img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.0
        img = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(img))
        return img

    def KeyNetAffNetHardNet(self, images: List[Path]):
        for im_path in images:
            img = self.load_torch_image(str(im_path)).to(self.device)
            keypts = KF.KeyNetAffNetHardNet(
                num_features=self.n_features, upright=True, device=torch.device("cuda")
            ).forward(img)
            self.kpts[im_path.stem] = keypts[0].cpu().detach().numpy()[-1, :, :, -1]
            self.descriptors[im_path.stem] = keypts[2].cpu().detach().numpy()[-1, :, :]

        return self.kpts, self.descriptors


def LocalFeatConfFile(cfg_edict) -> edict:
    local_feature = cfg_edict.LOCAL_FEAT_LOCAL_FEATURE

    if local_feature == "ALIKE":
        cfg_dict = edict(
            {
                "model": cfg_edict.LOCAL_FEAT_ALIKE_MODEL,
                "device": cfg_edict.LOCAL_FEAT_ALIKE_DEVICE,
                "top_k": cfg_edict.LOCAL_FEAT_N_FEATURES,
                "scores_th": cfg_edict.LOCAL_FEAT_ALIKE_SCORES_TH,
                "n_limit": cfg_edict.LOCAL_FEAT_ALIKE_N_LIMIT,
                "subpixel": cfg_edict.LOCAL_FEAT_ALIKE_SUBPIXEL,
            }
        )

    elif local_feature == "ORB":
        cfg_dict = edict(
            {
                "scaleFactor": cfg_edict.LOCAL_FEAT_ORB_SCALE_FACTOR,
                "nlevels": cfg_edict.LOCAL_FEAT_ORB_NLEVELS,
                "edgeThreshold": cfg_edict.LOCAL_FEAT_ORB_EDGE_THRESHOLD,
                "firstLevel": cfg_edict.LOCAL_FEAT_ORB_FIRST_LEVEL,
                "WTA_K": cfg_edict.LOCAL_FEAT_ORB_WTA_K,
                "scoreType": cfg_edict.LOCAL_FEAT_ORB_SCORE_TYPE,
                "patchSize": cfg_edict.LOCAL_FEAT_ORB_PATCH_SIZE,
                "fastThreshold": cfg_edict.LOCAL_FEAT_ORB_FAST_THRESHOLD,
            }
        )

    elif local_feature == "KeyNetAffNetHardNet":
        cfg_dict = edict({})

    elif local_feature == "RootSIFT":
        cfg_dict = edict({})

    else:
        print("In LocalFeatConfFile() in local_features.py missing local feat")
        quit()

    return cfg_dict


class LocalFeatureExtractor:
    def __init__(
        self,
        local_feature: str = "ORB",
        local_feature_cfg: dict = None,
        n_features: int = 1024,
        cam0_calib: str = "",
    ) -> None:
        self.local_feature = local_feature
        self.detector_and_descriptor = LocalFeatures(
            local_feature, n_features, local_feature_cfg
        )
        self.model1, self.width1, self.height1, other = cam0_calib.strip().split(",", 3)
        params1 = other.split(",", 7)
        self.params1 = np.array(params1).astype(np.float32)

    def run(self, database, keyframe_dir, image_format) -> None:
        db = db_colmap.COLMAPDatabase.connect(str(database))
        camera_id1 = db.add_camera(self.model1, self.width1, self.height1, self.params1)
        kfrms = os.listdir(keyframe_dir)
        kfrms.sort()
        existing_images = dict(
            (image_id, name)
            for image_id, name in db.execute("SELECT image_id, name FROM images")
        )

        for img in kfrms:
            if img not in existing_images.values():
                extract = getattr(self.detector_and_descriptor, self.local_feature)
                kpts, descriptors = extract([keyframe_dir / img])
                kp = kpts[img[: -len(image_format) - 1]]
                desc = descriptors[img[: -len(image_format) - 1]]

                kp = kp[:, 0:2]

                desc_len = np.shape(desc)[1]
                zero_matrix = np.zeros((np.shape(desc)[0], 128 - desc_len))
                desc = np.append(desc, zero_matrix, axis=1)
                desc.astype(np.float32)
                desc = np.absolute(desc)

                desc = desc * 512 / np.linalg.norm(desc, axis=1).reshape((-1, 1))
                desc = np.round(desc)
                desc = np.array(desc, dtype=np.uint8)

                one_matrix = np.ones((np.shape(kp)[0], 1))
                kp = np.append(kp, one_matrix, axis=1)
                zero_matrix = np.zeros((np.shape(kp)[0], 3))
                kp = np.append(kp, zero_matrix, axis=1).astype(np.float32)

                img_id = db.add_image(img, camera_id1)
                db.add_keypoints(img_id, kp)
                db.add_descriptors(img_id, desc)
                db.commit()

        db.close()
