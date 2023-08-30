import importlib
import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from easydict import EasyDict as edict

from lib import utils
from lib.utils.plot import make_traj_plot
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatures
from lib.matching import Matcher, make_match_plot
from lib.thirdparty.alike.alike import ALike, configs

MAX_WINDOW_SIZE = 1400

# TODO: use logger instead of print
logger = logging.getLogger(__name__)


# TODO: make ransac function independent from KeyFrameSelector class
def ransac(
    kpts1: np.ndarray, kpts2: np.ndarray, threshold: float = 4
) -> np.ndarray:
    pass


# TODO: integrate ALike in LocalFeatures (see self.extract_features() method)
class KeyFrameSelector:
    def __init__(
        self,
        keyframes_list: KeyFrameList,
        last_keyframe_pointer: int = 0,
        last_keyframe_delta: int = 0,
        keyframes_dir: Union[str, Path] = "colmap_imgs",
        kfs_method: str = "local_features",
        local_feature: str = "ORB",
        local_feature_cfg: dict = None,
        n_features: int = 512,
        kfs_matcher: str = "mnn_cosine",
        geometric_verification: str = "pydegensac",
        realtime_viz: bool = False,
        viz_res_path: Union[str, Path] = None,
        verbose: bool = False,
        innovation_threshold_pix: int = 5,
        min_matches: int = 5,
        error_threshold: int = 4,
        iterations: int = 1000,
        n_camera: int = 1,

    ) -> None:
        """
        __init__ _summary_

        Args:
            keyframes_list (List[Keyframe]): _description_
            last_keyframe_pointer (int): _description_
            last_keyframe_delta (int): _description_
            keyframes_dir (Union[str, Path], optional): _description_. Defaults to "colmap_imgs".
            kfs_method (Union[str, Path], optional): _description_. Defaults to "local_features".
            local_feature (str, optional): _description_. Defaults to "ORB".
            local_feature_cfg (dict, optional): _description_. Defaults to None.
            n_features (int, optional): _description_. Defaults to 512.
            kfs_matcher (str, optional): _description_. Defaults to "mnn_cosine".
            geometric_verification (str, optional): _description_. Defaults to "ransac".
            realtime_viz (bool, optional): _description_. Defaults to False.
            viz_res_path (Union[str, Path], optional): _description_. Defaults to None.
        """
        # TODO: validate input parameters
        self.keyframes_list = keyframes_list
        self.keyframes_dir = Path(keyframes_dir)
        self.keyframes_dir.mkdir(exist_ok=True, parents=True)
        self.method = kfs_method
        self.local_feature = local_feature
        self.local_feature_cfg = edict(local_feature_cfg)
        self.n_features = n_features
        self.matcher = kfs_matcher
        self.geometric_verification = geometric_verification
        self.innovation_threshold_pix=innovation_threshold_pix
        self.min_matches=min_matches
        self.error_threshold=error_threshold
        self.iterations=iterations
        self.n_camera=n_camera
        if n_camera > 1:
            for i in range(1, n_camera):
                camera_img_path = Path(keyframes_dir).parent / Path(f"cam{i}")
                camera_img_path.mkdir()

        self.realtime_viz = realtime_viz
        if self.realtime_viz:
            cv2.namedWindow("Keyframe Selection")

        if viz_res_path is not None:
            self.viz_res_path = Path(viz_res_path)
            if self.viz_res_path.exists():
                shutil.rmtree(self.viz_res_path)
            self.viz_res_path.mkdir()
        else:
            self.viz_res_path = None
        self.timer = None
        self.verbose = verbose

        # Set initial images to None
        self.img1 = None
        self.img2 = None

        # Initialize LocalFeature object
        if self.method == "local_features":
            self.feature_extractor = LocalFeatures(
                method=self.local_feature,
                n_features=self.n_features,
                cfg=self.local_feature_cfg,
            )

        # Set initial keyframes, descr, mpts to None
        self.last_keyframe_features = None
        self.kpts1 = None
        self.kpts2 = None
        self.desc1 = None
        self.desc2 = None
        self.mpts1 = None
        self.mpts2 = None
        self.median_match_dist = None

        # Set initial pointer and delta
        self.pointer = last_keyframe_pointer
        self.delta = last_keyframe_delta

    def clear_matches(self) -> None:
        # NOTE: Now it is executed for safety reasons by self.run() at the end of the keyframeselection process... it may be removed if we want to keep track of the previous matched
        self.kpts1 = None
        self.kpts2 = None
        self.desc1 = None
        self.desc2 = None
        self.mpts1 = None
        self.mpts2 = None
        self.median_match_dist = None

    def extract_features(self, img1: Union[str, Path], img2: Union[str, Path]) -> bool:
        # TODO: to speed up the procedure, keep the keypoints and descriptors of the last keyframe in memory, instead of extracting them again

        self.img1 = Path(img1)
        self.img2 = Path(img2)

        if self.local_feature == "ORB":
            all_keypoints, all_descriptors, lafs = self.feature_extractor.ORB([img1, img2])
            self.kpts1 = all_keypoints[img1.stem][:, 0:2]
            self.kpts2 = all_keypoints[img2.stem][:, 0:2]
            self.desc1 = all_descriptors[img1.stem][:, 0:32]
            self.desc2 = all_descriptors[img2.stem][:, 0:32]

        elif self.local_feature == "ALIKE":

            all_keypoints, all_descriptors, lafs = self.feature_extractor.ALIKE([img1, img2])
            self.kpts1 = all_keypoints[img1.stem]
            self.kpts2 = all_keypoints[img2.stem]
            self.desc1 = all_descriptors[img1.stem]
            self.desc2 = all_descriptors[img2.stem]

        elif self.local_feature == "KeyNetAffNetHardNet":

            all_keypoints, all_descriptors, lafs = self.feature_extractor.KeyNetAffNetHardNet([img1, img2])
            self.kpts1 = all_keypoints[img1.stem]
            self.kpts2 = all_keypoints[img2.stem]
            self.desc1 = all_descriptors[img1.stem]
            self.desc2 = all_descriptors[img2.stem]

        else:
            # Here we can implement methods like LoFTR
            logger.error("Error! Only local_features method is implemented")
            quit()

        if self.timer is not None:
            self.timer.update("features extraction")

        return True

    def match_features(self) -> bool:
        assert all(
            [self.kpts1 is not None, self.kpts2 is not None]
        ), "kpts1 or kpts2 is None, run extract_features first"

        # Here we should handle that we can use different kinds of matcher (also adding the option in config file)
        if self.matcher == "mnn_cosine":
            matcher = Matcher(self.desc1, self.desc2)
            matches = matcher.mnn_matcher_cosine()
            matches_im1 = matches[:, 0]
            matches_im2 = matches[:, 1]
            self.mpts1 = self.kpts1[matches_im1]
            self.mpts2 = self.kpts2[matches_im2]

        else:
            # Here we can implement matching methods
            logger.error("Error! Only mnn_cosine method is implemented")
            quit()

        ### Ransac to eliminate outliers
        mask = np.ones(len(self.mpts1), dtype=bool)
        if self.geometric_verification == "pydegensac":
            try:
                pydegensac = importlib.import_module("pydegensac")
                geometric_verification = "pydegensac"
            except:
                geometric_verification == "ransac"
                logging.error("Pydegensac not available.")
        elif self.geometric_verification == "ransac":
            geometric_verification = "ransac"

        if geometric_verification == "pydegensac":
            try:
                F, mask = pydegensac.findFundamentalMatrix(
                    self.mpts1,
                    self.mpts2,
                    px_th=self.error_threshold,
                    conf=0.999,
                    max_iters=self.iterations,
                    laf_consistensy_coef=-1,
                    error_type="sampson",
                    symmetric_error_check=True,
                    enable_degeneracy_check=True,
                )
                logger.info(f"Pydegensac found {mask.sum()}/{len(mask)} inliers")
            except:
                return False
        
        elif geometric_verification == "ransac":
            # TODO: move RANSAC to a separate function
            match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
            rands = []
            scores = []
            for i in range(self.iterations):
                rand = random.randrange(0, self.mpts1.shape[0])
                reference_distance = np.linalg.norm(self.mpts1[rand] - self.mpts2[rand])
                score = np.sum(
                    np.absolute(match_dist - reference_distance) < self.error_threshold
                ) / len(match_dist)
                rands.append(rand)
                scores.append(score)
            max_consensus = rands[np.argmax(scores)]
            reference_distance = np.linalg.norm(
                self.mpts1[max_consensus] - self.mpts2[max_consensus]
            )
            mask = np.absolute(match_dist - reference_distance) > self.error_threshold
            logger.info(f"Ransac found {mask.sum()}/{len(mask)} inliers")

        else:
            # Here we can implement other methods
            logger.error(
                f"Invalid choise for outlier rejection. Only Pydegensac (external library) and Ransac are implemented"
            )
            quit()
        self.mpts1 = self.mpts1[mask, :]
        self.mpts2 = self.mpts2[mask, :]

        if self.timer is not None:
            self.timer.update("matching")
        return True

    def innovation_check(self) -> bool:
        match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
        self.median_match_dist = np.median(match_dist)
        logger.info(f"median_match_dist: {self.median_match_dist:.2f}")

        if len(self.mpts1) < self.min_matches:
            logger.info("Frame rejected: not enogh matches")
            self.delta += 1
            if self.timer is not None:
                self.timer.update("innovation check")

            return False
        
        if self.median_match_dist > self.innovation_threshold_pix:
            existing_keyframe_number = len(os.listdir(self.keyframes_dir))
            for c in range(self.n_camera):
                current_img = self.img2.name
                imgs_folder = self.img2.parent.parent
                shutil.copy(
                    imgs_folder / f"cam{c}" / current_img,
                    self.keyframes_dir.parent / f"cam{c}" / f"{utils.Id2name(existing_keyframe_number)}",
                )
            camera_id = 1

            new_keyframe = KeyFrame(
                self.img2,
                existing_keyframe_number,
                utils.Id2name(existing_keyframe_number),
                camera_id,
                self.pointer + self.delta + 1,
            )
            logger.info('ok3')
            self.keyframes_list.add_keyframe(new_keyframe)
            logger.info('ok4')
            # self.keyframes_list.append(new_keyframe)
            #logger.info(
            #    f"Frame accepted. New_keyframe image_id: {new_keyframe.image_id}"
            #)

            self.pointer += 1 + self.delta
            self.delta = 0
            if self.timer is not None:
                self.timer.update("innovation check")

            return True

        else:
            logger.info("Frame rejected")
            self.delta += 1
            if self.timer is not None:
                self.timer.update("innovation check")

            return False

    def run(self, img1: Union[str, Path], img2: Union[str, Path]):
        self.timer = utils.AverageTimer()

        read = False
        while read == False:
            try:
                i1 = cv2.cvtColor(cv2.imread(str(img1)), cv2.COLOR_BGR2RGB)
                i2 = cv2.cvtColor(cv2.imread(str(img2)), cv2.COLOR_BGR2RGB)
                read = True
            except:
                print("Failed reading images (opencv). Trying again ..")
                

        #try:
        #    if not self.extract_features(img1, img2):
        #        raise RuntimeError("Error in extract_features")
        #    if not self.match_features():
        #        raise RuntimeError("Error in match_features")
        #    keyframe_accepted = self.innovation_check()
        self.extract_features(img1, img2)
        self.match_features()
        keyframe_accepted = self.innovation_check()

        #except RuntimeError as e:
        #    keyframe_accepted = False
        #    logger.error(e)
        #    return self.keyframes_list, self.pointer, self.delta, None

        if self.viz_res_path is not None or self.realtime_viz:
            img = cv2.imread(str(self.img2), cv2.IMREAD_UNCHANGED)
            match_img = make_match_plot(img, self.mpts1, self.mpts2)
            traj_img = make_traj_plot('./keyframes.pkl', './points3D.pkl', match_img.shape[1], match_img.shape[0])
            if keyframe_accepted:
                win_name = f"{self.local_feature} - MMD {self.median_match_dist:.2f}: Keyframe accepted"
            else:
                win_name = f"{self.local_feature} - MMD {self.median_match_dist:.2f}: Frame rejected"

        if self.realtime_viz:
            cv2.setWindowTitle("Keyframe Selection", win_name)
            #cv2.imshow("Keyframe Selection", traj_img)
            conc = np.concatenate((match_img, traj_img), axis=1)
            height, width, channels = conc.shape
            conc = cv2.resize(conc, (MAX_WINDOW_SIZE, int(height*MAX_WINDOW_SIZE/width)))
            cv2.imshow("Keyframe Selection", conc)
            if cv2.waitKey(1) == ord("q"):
                sys.exit()

        if self.viz_res_path is not None:
            #out_name = f"{self.img1.stem}_{win_name}.jpg"
            out_name = f"{self.img1.stem}.jpg"
            cv2.imwrite(str(self.viz_res_path / out_name), conc)

        self.clear_matches()

        time = self.timer.print("Keyframe Selection")

        return self.keyframes_list, self.pointer, self.delta, time


def KeyFrameSelConfFile(cfg_edict) -> edict:
    local_feature = cfg_edict.KFS_LOCAL_FEATURE

    if local_feature == 'ALIKE':
        cfg_dict = edict(
            {
                "model": cfg_edict.ALIKE_MODEL,
                "device": cfg_edict.ALIKE_DEVICE,
                "top_k": cfg_edict.KFS_N_FEATURES,
                "scores_th": cfg_edict.ALIKE_SCORES_TH,
                "n_limit": cfg_edict.ALIKE_N_LIMIT,
                "subpixel": cfg_edict.ALIKE_SUBPIXEL,
            }
        )

    elif local_feature == 'ORB':
        cfg_dict = edict(
            {
                "scaleFactor": cfg_edict.ORB_SCALE_FACTOR,
                "nlevels": cfg_edict.ORB_NLEVELS,
                "edgeThreshold": cfg_edict.ORB_EDGE_THRESHOLD,
                "firstLevel": cfg_edict.ORB_FIRST_LEVEL,
                "WTA_K": cfg_edict.ORB_WTA_K,
                "scoreType": cfg_edict.ORB_SCORE_TYPE,
                "patchSize": cfg_edict.ORB_PATCH_SIZE,
                "fastThreshold": cfg_edict.ORB_FAST_THRESHOLD,
            }
        )
    
    elif local_feature == 'KeyNetAffNetHardNet':
        cfg_dict = edict(
            {

            }
        )
    return cfg_dict


if __name__ == "__main__":

    image_path = Path("res/ponte_test/img")
    im_ext = ".JPG"
    out_dir = Path("res/ponte_test/keyframes")

    img_list = sorted(image_path.glob(f"*{im_ext}"))
    out_dir.mkdir(exist_ok=True)

    keyframes_list = KeyFrameList()
    pointer = 0  # pointer points to the last oriented image
    delta = 0  # delta is equal to the number of processed but not oriented imgs

    alike_cfg = edict(
        {
            "model": "alike-t",
            "device": "cuda",
            "top_k": 1024,
            "scores_th": 0.2,
            "n_limit": 5000,
            "subpixel": False,
        }
    )
    keyframe_selector = KeyFrameSelector(
        keyframes_list=keyframes_list,
        last_keyframe_pointer=pointer,
        last_keyframe_delta=delta,
        keyframes_dir=out_dir,
        kfs_method="local_features",
        geometric_verification="pydegensac",
        local_feature="ALIKE",
        local_feature_cfg=alike_cfg,
        n_features=1024,
        realtime_viz=True,
        viz_res_path=out_dir / "viz_res",
    )

    img0 = img_list[0]
    camera_id = 1
    image_id = 0
    shutil.copy(
        img0,
        out_dir / f"{img0.name}",
    )
    keyframes_list.add_keyframe(
        KeyFrame(
            img0,
            {img0.name},
            {img0.name},
            camera_id,
            image_id,
        )
    )

    for i, cur_frame in enumerate(img_list[1:]):
        print(f"Processing {i} of {len(img_list) - 1}")

        last_keyframe = img_list[pointer]
        (
            keyframes_list,
            pointer,
            delta,
            dt,
        ) = keyframe_selector.run(last_keyframe, cur_frame)

    print("Done")
