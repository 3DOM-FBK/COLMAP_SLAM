import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from easydict import EasyDict as edict

from lib import db_colmap, cameras
from lib.thirdparty.alike.alike import ALike, configs

from lib.thirdparty.SuperGlue.models.matching import Matching
from lib.thirdparty.LightGlue.lightglue import SuperPoint
from lib.thirdparty.LightGlue.lightglue.utils import load_image

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

        elif self.method == "DISK":
            self.orb_cfg = cfg
            self.device = torch.device("cuda")  # TODO set to cuda
            self.disk = KF.DISK.from_pretrained('depth').to(self.device)

        elif self.method == "KeyNetAffNetHardNet":
            self.kornia_cfg = cfg
            self.device = torch.device("cuda")  # TODO set to cuda

        elif self.method == "SuperPoint":
            self.kornia_cfg = cfg

        elif self.method == "SuperGlue":
            self.superglue_cfg = cfg
            device = 'cuda'
            config = {
                'superpoint': {
                    'nms_radius': cfg.nms_radius,
                    'keypoint_threshold': cfg.keypoint_threshold,
                    'max_keypoints': n_features
                },
                'superglue': {
                    'weights': cfg.weights,
                    'sinkhorn_iterations': cfg.sinkhorn_iterations,
                    'match_threshold': cfg.match_threshold,
                }
            }
            matching = Matching(config).eval().to(device)
            self.matcher = matching

        elif self.method == "LoFTR":
            self.kornia_cfg = cfg

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

            laf = None

        return self.kpts, self.descriptors, laf

    def ALIKE(self, images: List[Path]):
        for im_path in images:
            img = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB)
            #h, w, c = img.shape
            #ratio = 0.15
            #img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
            features = self.model(img, sub_pixel=self.alike_cfg.subpixel)

            self.kpts[im_path.stem] = features["keypoints"] #* 1/ratio
            self.descriptors[im_path.stem] = features["descriptors"] #* 1/ratio

            laf = None

        return self.kpts, self.descriptors, laf

    def load_torch_image(self, fname):
        cv_img = cv2.imread(fname)
        img = K.image_to_tensor(cv_img, False).float() / 255.0
        img = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(img))
        return img

    def load_torch_image_rgb(self, fname):
        cv_img = cv2.imread(fname)
        img = K.image_to_tensor(cv_img, False).float() / 255.0
        return img

    def DISK(self, images: List[Path]):
        # Inspired by: https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        disk = self.disk
        with torch.inference_mode():
            for im_path in images:
                img = self.load_torch_image_rgb(str(im_path)).to(self.device)
                features = disk(img, self.n_features, pad_if_not_divisible=True)[0]
                kps1, descs = features.keypoints, features.descriptors

                self.kpts[im_path.stem] = kps1.cpu().detach().numpy()
                self.descriptors[im_path.stem] = descs.cpu().detach().numpy()

                laf = None

        return self.kpts, self.descriptors, laf

    def SuperPoint(self, images: List[Path]):
        with torch.inference_mode():
            for im_path in images:
                extractor = SuperPoint(max_num_keypoints=self.n_features).eval().cuda()
                image = load_image(im_path).cuda()
                feats = extractor.extract(image)
                kpt = feats['keypoints'].cpu().detach().numpy()
                desc = feats['descriptors'].cpu().detach().numpy()
                self.kpts[im_path.stem] = kpt.reshape(-1, kpt.shape[-1])
                self.descriptors[im_path.stem] = desc.reshape(-1, desc.shape[-1])

                #print(self.kpts[im_path.stem].shape)
                #print(self.descriptors[im_path.stem].shape)

                laf = None

        return self.kpts, self.descriptors, laf

    def KeyNetAffNetHardNet(self, images: List[Path]):
        for im_path in images:
            img = self.load_torch_image(str(im_path)).to(self.device)
            keypts = KF.KeyNetAffNetHardNet(
                num_features=self.n_features, upright=True, device=torch.device("cuda")
            ).forward(img)
            laf = keypts[0].cpu().detach().numpy()
            self.kpts[im_path.stem] = keypts[0].cpu().detach().numpy()[-1, :, :, -1]
            self.descriptors[im_path.stem] = keypts[2].cpu().detach().numpy()[-1, :, :]

        return self.kpts, self.descriptors, laf

    def SuperGlue(self, images: List[Path]):
        self.kpts = {}
        self.descriptors = {}
        laf = None

        return self.kpts, self.descriptors, laf

    def LoFTR(self, images: List[Path]):
        self.kpts = {}
        self.descriptors = {}
        laf = None

        return self.kpts, self.descriptors, laf


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

    elif local_feature == "SuperPoint":
        cfg_dict = edict({})

    elif local_feature == "DISK":
        cfg_dict = edict({})

    elif local_feature == "LoFTR":
        cfg_dict = edict({})

    elif local_feature == "SuperGlue":
        cfg_dict = edict(
            {
            'nms_radius': cfg_edict.SUPERGLUE_NMS_RADIUS,
            'keypoint_threshold': cfg_edict.SUPERGLUE_KEYPOINT_THRESHOLD,
            'weights': cfg_edict.SUPERGLUE_WEIGHTS,
            'sinkhorn_iterations': cfg_edict.SUPERGLUE_SINKHORN_ITERATIONS,
            'match_threshold': cfg_edict.SUPERGLUE_MATCH_THRESHOLD,
            }
        )

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
        cam_calib : dict = {},
        database : Union[str, Path] = "",
    ) -> None:
        self.local_feature = local_feature
        self.detector_and_descriptor = LocalFeatures(
            local_feature, n_features, local_feature_cfg
        )

        cameras.CreateCameras(cam_calib, database)


    def run(self, database, keyframes_dict, kfrms, all_frames_dir, keyframe_dir, image_format, kpts_key_colmap_id, descs_key_colmap_id, laf_key_colmap_id) -> None:

        kfm_batch = []
        db = db_colmap.COLMAPDatabase.connect(str(database))
        cams = os.listdir(keyframe_dir)

        #kfrm_objs = keyframes_list.keyframes()

        existing_images = dict(
            (image_id, name)
            for image_id, name in db.execute("SELECT image_id, name FROM images")
        )

        for cam in cams:
            #imgs = os.listdir(keyframe_dir / cam)
            #imgs = [kf.keyframe_name() for kf in kfrm_objs]
            imgs = kfrms
            print()
            print()
            print(imgs)
            for img in imgs:
                img = Path(cam) / Path(img)
                if str(img.parent) + "/" + str(img.name) not in existing_images.values():
                    #kfm = keyframes_list.get_keyframe_by_name(img.name) 
                    if cam == "cam0":
                        #kfm_batch.append(kfm.image_name().name)
                        #id, ext = img.name.split('.', 1)
                        #image_name = Path(keyframes_dict[int(id)]['image_name']).name
                        image_name = Path(keyframes_dict[img.name]['image_name']).name
                        kfm_batch.append(image_name)
                        
                    shutil.copy(
                        Path(f"./imgs/{cam}") / image_name,
                        keyframe_dir / img,
                    )

                    if self.local_feature == "SuperGlue" or self.local_feature == "LoFTR":
                        img_id = db.add_image(str(img.parent) + "/" + str(img.name), 1)
                        db.commit()
                    else:
                        extract = getattr(self.detector_and_descriptor, self.local_feature)
                        kpts, descriptors, laf = extract([keyframe_dir / img])
                        kp = kpts[img.name[: -len(image_format) - 1]]
                        desc = descriptors[img.name[: -len(image_format) - 1]]

                        img_id = db.add_image(str(img.parent) + "/" + str(img.name), 1)
                        kpts_key_colmap_id[img_id] = kp
                        descs_key_colmap_id[img_id] = desc
                        laf_key_colmap_id[img_id] = laf
 
                        kp = kp[:, 0:2]
    
                        desc_len = np.shape(desc)[1]
                        if self.local_feature != 'SuperPoint':
                            zero_matrix = np.zeros((np.shape(desc)[0], 128 - desc_len))
                            desc = np.append(desc, zero_matrix, axis=1)
                        else:
                            desc = desc[:, :128]
                        desc.astype(np.float32)
                        desc = np.absolute(desc)
    
                        desc = desc * 512 / np.linalg.norm(desc, axis=1).reshape((-1, 1))
                        desc = np.round(desc)
                        desc = np.array(desc, dtype=np.uint8)
    
                        one_matrix = np.ones((np.shape(kp)[0], 1))
                        kp = np.append(kp, one_matrix, axis=1)
                        zero_matrix = np.zeros((np.shape(kp)[0], 3))
                        kp = np.append(kp, zero_matrix, axis=1).astype(np.float32)
    
                        
                        db.add_keypoints(img_id, kp)
                        db.add_descriptors(img_id, desc)
                        db.commit()

        db.close()

        return kpts_key_colmap_id, descs_key_colmap_id, laf_key_colmap_id, kfm_batch
