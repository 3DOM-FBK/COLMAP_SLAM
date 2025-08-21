import os
import cv2
import time
import torch
import shutil
import numpy as np
import kornia.feature as KF

from typing import List, Tuple, Dict
from pyquaternion import Quaternion
from pathlib import Path

from src.odometry.local_features import LocalFeatures
from src.odometry.db_colmap import COLMAPDatabase
from src.odometry.custom_incremental_pipeline import reconstruct

import pycolmap
from pycolmap import Database, Camera, Image, ListPoint2D, Rigid3d, Rotation3d, TwoViewGeometry, logging


def quat(colmap_quat: np.ndarray) -> Quaternion:
    x, y, z, w = colmap_quat
    return Quaternion(np.array([w, x, y, z]))


class VisualOdometry:
    """
    Optimized version focusing on:
    - avoiding deep copies and redundant allocations
    - minimizing GPU<->CPU transfers
    - in-place dict updates
    - caching LightGlue LAFs per image
    - reducing repeated DB reloads
    - minor micro-opts and small bug fixes
    """

    def __init__(
        self,
        working_dir: Path,
        config: dict,
        camera_config: dict,
    ) -> None:
        logging.verbose_level = 0
        logging.minloglevel = 2

        # --- state ---
        self.snapshot_count = 0
        self.keyframes_names: Dict[str, int] = {}
        self.keyframes_ids: Dict[int, str] = {}
        self.keyframes_master_ids: List[int] = []
        self.config = config
        self.camera_config = camera_config
        self.baseline = config['mapping']['baseline']
        self.log = config['general']['log']
        self.rig_match_rule = config['mapping']['rig_match_rule']
        self.height, self.width = camera_config['cam0']['height'], camera_config['cam0']['width']
        self.N_reinit = 0
        self.run_BA = False  # control BA only after last stereo image is added
        self.control_params: Dict = {}
        self.log_data: Dict = {}

        self.images_dir = working_dir / "images"

        self.test = self.config['general']['test']
        self.cameras = sorted(self.config['mapping']['cameras'], key=lambda x: int(x[3:]))
        self.n_cameras = len(self.cameras)
        self.cameras_for_baseline_estim = config['mapping']['cameras_for_baseline_estim']
        if "cam0" not in self.cameras_for_baseline_estim:
            raise ValueError("cam0 must be included in cameras_for_baseline_estim")
        # pick a second camera for scale estimation
        self.second_baseline_camera = next((c for c in self.cameras_for_baseline_estim if c != "cam0"), "cam1" if self.n_cameras > 1 else "cam0")

        # features storage
        self.keypoints: Dict[str, torch.Tensor] = {}
        self.descriptors: Dict[str, torch.Tensor] = {}
        self.lafs_cache: Dict[str, torch.Tensor] = {}  # cached LAFs to avoid recomputation

        self.database_path = working_dir / "database.db"
        if self.database_path.exists():
            self.database_path.unlink()

        self.out_dir = working_dir / "out"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.images: List[str] = []

        # Local feature extractor + LightGlue
        self.local_features = LocalFeatures(
            self.width,
            self.height,
            config['local_features'],
            self.log,
        )
        if config['local_features']['features_name'] == "aliked":
            self.lightglue_model = "aliked"
        elif config['local_features']['features_name'] == "superpoint":
            self.lightglue_model = "superpoint"
        else:
            raise ValueError("Invalid local features model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lg_matcher = KF.LightGlueMatcher(self.lightglue_model).eval().to(self.device)

        # Initialize database and odometry variables
        self.baseline_old = 0.0
        self.keyframe_count = 1
        self.keyframe_id = 1
        self.t_cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.q_cumulative = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
        self.db = Database(str(self.database_path))
        self.db_dirty = False  # track when DB has new data not yet loaded into controller

        # Mapper options
        self.sliding_window = self.config['mapping']['sliding_window']
        pycolmap.set_random_seed(0)
        self.options = pycolmap.IncrementalPipelineOptions()
        self.options.ba_refine_focal_length = False
        self.options.ba_refine_principal_point = False
        self.options.ba_refine_extra_params = False
        self.options.extract_colors = False
        self.options.fix_existing_images = False
        self.options.ba_global_max_num_iterations = 25
        self.options.ba_global_max_refinements = 5
        self.options.multiple_models = False
        # NOTE: The original set init_image_id1 twice; keeping defaults here
        # self.options.init_image_id1 = 1
        # self.options.init_image_id2 = 2

        self.reconstruction_manager = pycolmap.ReconstructionManager()
        self.controller = pycolmap.IncrementalPipeline(
            self.options, str(self.images_dir), str(self.database_path), self.reconstruction_manager
        )

        self.mapper_options = self.controller.options.get_mapper()
        self.mapper_options.init_max_forward_motion = 0.99
        self.mapper_options.init_min_tri_angle = 1.0
        self.mapper_options.init_max_error = 100.0
        self.mapper_options.abs_pose_max_error = 50.0
        self.mapper_options.abs_pose_min_num_inliers = 8
        self.mapper_options.abs_pose_min_inlier_ratio = 0.01
        self.mapper_options.filter_max_reproj_error = 1.5

        # Precompute image size arrays (avoid realloc in matcher)
        self.hw_np = np.array([self.height, self.width])

        # for LightGlue: constant scale/orientation LAF base (ones) cached per image len
        self._ones_cache: Dict[int, torch.Tensor] = {}

    # -------------------- utils --------------------
    def _laf_from_kps_cached(self, name: str, kps: torch.Tensor) -> torch.Tensor:
        """Cache LAF tensors per image to avoid recomputation in matching."""
        if name in self.lafs_cache:
            return self.lafs_cache[name]
        n = kps.shape[0]
        if n not in self._ones_cache:
            self._ones_cache[n] = torch.ones(1, n, 1, 1, device=self.device)
        lafs = KF.laf_from_center_scale_ori(kps[None], self._ones_cache[n])
        self.lafs_cache[name] = lafs
        return lafs

    def rig_match_pairs(self, img_name: str) -> List[Tuple[str, str]]:
        return [(f"{c1}/{img_name}", f"{c2}/{img_name}") for c1, c2 in self.rig_match_rule]

    def check_stereo(self, img1: pycolmap.Image, img2: pycolmap.Image) -> None:
        if img1.name.split("/")[1] != img2.name.split("/")[1]:
            raise RuntimeError("Stereo timestamps mismatch")

    def make_match_plot(
        self, img: np.ndarray, img2: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray, method: str = "flow",
    ) -> np.ndarray:
        # DEBUG ONLY (slow). Prefer disabling in production.
        if method == "flow":
            match_img = img.copy()
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), 3, lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)
        else:  # "pair"
            img1_w = img.shape[1]
            match_img = np.concatenate((img, img2), axis=1)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0] + img1_w)), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(match_img, p1, 1, (0, 0, 255), 3, lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), 3, lineType=16)
        return match_img

    @torch.inference_mode()
    def match_features(self, keypoints: Dict[str, torch.Tensor], descriptors: Dict[str, torch.Tensor], pairs: List[Tuple[str, str]]):
        matches = {}
        for img1, img2 in pairs:
            kps1 = keypoints[img1]
            kps2 = keypoints[img2]
            descs1 = descriptors[img1]
            descs2 = descriptors[img2]

            # ensure on device once
            if kps1.device != self.device:
                kps1 = kps1.to(self.device, non_blocking=True)
                keypoints[img1] = kps1
            if kps2.device != self.device:
                kps2 = kps2.to(self.device, non_blocking=True)
                keypoints[img2] = kps2
            if descs1.device != self.device:
                descs1 = descs1.to(self.device, non_blocking=True)
                descriptors[img1] = descs1
            if descs2.device != self.device:
                descs2 = descs2.to(self.device, non_blocking=True)
                descriptors[img2] = descs2

            lafs1 = self._laf_from_kps_cached(img1, kps1)
            lafs2 = self._laf_from_kps_cached(img2, kps2)

            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=self.hw_np, hw2=self.hw_np)
            matches[(img1, img2)] = idxs  # keep on device
        return matches

    @torch.inference_mode()
    def match_distance(self, keypoints: Dict[str, torch.Tensor], matches: Dict[Tuple[str, str], torch.Tensor], keyframe_name: str, frame_name: str) -> float:
        # compute on GPU to avoid transfers
        idxs = matches[(keyframe_name, frame_name)]
        mpts1 = keypoints[keyframe_name][idxs[:, 0]]
        mpts2 = keypoints[frame_name][idxs[:, 1]]
        # Torch median of L2 distances
        match_dist = torch.linalg.norm(mpts1 - mpts2, dim=1)
        median_match_dist = torch.median(match_dist).item()
        return float(median_match_dist)

    def write_keypoints_to_db(self, db: COLMAPDatabase, keyframe_name: str, image_id: int, camera_id: int, keypoints: Dict[str, torch.Tensor]) -> None:
        image = Image(
            name=keyframe_name,
            points2D=ListPoint2D(np.empty((0, 2), dtype=np.float64)),
            cam_from_world=Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0]),
            camera_id=camera_id,
            id=image_id,
        )
        db.write_image(image, use_image_id=True)
        db.write_keypoints(image_id=image_id, keypoints=keypoints[keyframe_name].detach().cpu().numpy())
        self.db_dirty = True

    def _extract_and_store(self, name: str, img: np.ndarray) -> None:
        new_kps, new_descs = self.local_features.extract(name, img)
        # Keep tensors on device to avoid later copies
        for k, v in new_kps.items():
            new_kps[k] = v.to(self.device, non_blocking=True)
        for k, v in new_descs.items():
            new_descs[k] = v.to(self.device, non_blocking=True)
        self.keypoints.update(new_kps)
        self.descriptors.update(new_descs)
        # invalidate LAF cache if this name exists
        for k in new_kps.keys():
            self.lafs_cache.pop(k, None)

    def reinitialize(self) -> None:
        if self.log:
            print('[CSLAM:] Reinitializing..')
        self.baseline_old = 0.0
        self.keyframe_count = 1
        self.keyframe_id = 1
        self.t_cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.q_cumulative = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))

        # reset on-disk DB
        if self.database_path.exists():
            self.database_path.unlink()
        self.db = Database(str(self.database_path))
        self.db_dirty = False

        # reset outputs
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.images.clear()
        self.N_reinit += 1
        self.keyframes_names.clear()
        self.keyframes_ids.clear()
        self.keyframes_master_ids.clear()
        self.keypoints.clear()
        self.descriptors.clear()
        self.lafs_cache.clear()
        self._ones_cache.clear()

        self.reconstruction_manager = pycolmap.ReconstructionManager()
        self.controller = pycolmap.IncrementalPipeline(
            self.options, str(self.images_dir), str(self.database_path), self.reconstruction_manager
        )

    def _maybe_load_db(self):
        if self.db_dirty:
            self.controller.load_database()
            self.db_dirty = False

    def run(
        self, image: str,
        images: List[np.ndarray],
        reinitialize: bool,
    ):
        self.images.append(image)

        if reinitialize:
            self.reinitialize()
            return [[image, None, None, None, None, None]], self.control_params, self.log_data

        if len(self.images) == 1:
            # first batch: register all cameras for the first timestamp
            for c, cam in enumerate(self.cameras):
                camera = Camera(self.camera_config[f"{cam}"])
                self.db.write_camera(camera)
                keyframe_name = f"{cam}/{image}"
                camera_id = 1 + c
                image_id = 1 + c
                self._extract_and_store(keyframe_name, images[c])
                self.write_keypoints_to_db(self.db, keyframe_name, image_id, camera_id, self.keypoints)
                self.keyframes_names[keyframe_name] = image_id
                self.keyframes_ids[image_id] = keyframe_name

            self.keyframe_name = f"cam0/{image}"
            self.keyframes_master_ids.append(1)

            if self.n_cameras != 1:
                pairs = self.rig_match_pairs(image)
                matches = self.match_features(self.keypoints, self.descriptors, pairs)
                for pair in pairs:
                    kfrm1, kfrm2 = pair
                    inlier_matches = matches[pair].detach().cpu().numpy()
                    self.db.write_two_view_geometry(
                        self.keyframes_names[kfrm1],
                        self.keyframes_names[kfrm2],
                        TwoViewGeometry({"inlier_matches": inlier_matches})
                    )
                self.db_dirty = True
            return [[image, None, None, None, None, None]], self.control_params, self.log_data

        # -------- Keyframe selection based on feature motion --------
        if self.log:
            t0 = time.time()
        frame_name = f"cam0/{image}"
        self._extract_and_store(frame_name, images[0])

        pairs = [(self.keyframe_name, frame_name)]
        matches = self.match_features(self.keypoints, self.descriptors, pairs)
        median_match_dist = self.match_distance(self.keypoints, matches, self.keyframe_name, frame_name)

        if median_match_dist < self.config['mapping']['max_match_distance']:
            # not a keyframe: free memory of temporary frame
            self.keypoints.pop(frame_name, None)
            self.descriptors.pop(frame_name, None)
            self.lafs_cache.pop(frame_name, None)
            return [[image, None, None, None, None, None]], self.control_params, self.log_data

        # --- Promote to keyframe on master camera ---
        inlier_matches = matches[(self.keyframe_name, frame_name)].detach().cpu().numpy()
        self.keyframe_count += 1
        self.keyframe_id += 1 * self.n_cameras
        self.keyframe_name = frame_name
        camera_id = 1
        self.write_keypoints_to_db(self.db, self.keyframe_name, self.keyframe_id, camera_id, self.keypoints)
        self.db.write_two_view_geometry(
            self.keyframe_id - 1 * self.n_cameras,
            self.keyframe_id,
            TwoViewGeometry({"inlier_matches": inlier_matches})
        )
        self.db_dirty = True
        self.keyframes_names[self.keyframe_name] = self.keyframe_id
        self.keyframes_ids[self.keyframe_id] = self.keyframe_name
        self.keyframes_master_ids.append(self.keyframe_id)

        # --- Match slave cameras ---
        if self.n_cameras != 1:
            for c, cam in enumerate(self.cameras):
                if c == 0:
                    continue
                slave_name = f"{cam}/{image}"
                slave_id = self.keyframe_id + 1 * c
                self.keyframes_names[slave_name] = slave_id
                self.keyframes_ids[slave_id] = slave_name
                camera_id = c + 1
                self._extract_and_store(slave_name, images[c])
                self.write_keypoints_to_db(self.db, slave_name, slave_id, camera_id, self.keypoints)

            pairs = self.rig_match_pairs(image)
            matches = self.match_features(self.keypoints, self.descriptors, pairs)
            for pair in pairs:
                kfrm1, kfrm2 = pair
                inlier_matches = matches[pair].detach().cpu().numpy()
                self.db.write_two_view_geometry(
                    self.keyframes_names[kfrm1],
                    self.keyframes_names[kfrm2],
                    TwoViewGeometry({"inlier_matches": inlier_matches})
                )
            self.db_dirty = True

        if self.log:
            t1 = time.time()
            print(f"[CSLAM] Matching time {t1 - t0:.2f} seconds")

        # --- Orientation / Reconstruction step ---
        if self.log:
            t0 = time.time()

        if 5 < self.keyframe_count < self.sliding_window:
            self._maybe_load_db()
            try:
                if self.config['mapping']['method'] == 'custom':
                    reconstruct(self.controller, self.mapper_options, self.keyframe_id, False, run_BA=True)
                else:
                    self.controller.reconstruct(self.mapper_options)
            except Exception:
                self.reinitialize()
                return [[image, None, None, None, None, None]], self.control_params, self.log_data
            if self.test:
                self.reconstruction_manager.write(self.out_dir)
            return [[image, None, None, None, None, None]], self.control_params, self.log_data

        elif self.keyframe_count >= self.sliding_window:
            self._maybe_load_db()

            if self.config['mapping']['method'] == 'custom':
                # Add new keyframes (entire rig)
                for c in range(self.n_cameras):
                    try:
                        if c == self.n_cameras - 1:
                            self.run_BA = True
                        if self.log:
                            tt0 = time.time()
                        reconstruct(self.controller, self.mapper_options, self.keyframe_id + c, True, run_BA=self.run_BA)
                        if self.log:
                            tt1 = time.time()
                            print('[CSLAM] Time for reconstruct:', tt1 - tt0)
                        self.run_BA = False
                    except Exception:
                        if self.log:
                            print('[CSLAM] Error in keyframe orientation')
                        self.reinitialize()
                        return [[image, None, None, None, None, None]], self.control_params, self.log_data

                # Sliding window: deregister oldest
                reconstruction = self.reconstruction_manager.get(idx=0)
                reg_image_ids = reconstruction.reg_image_ids()
                n_to_deregister = len(reg_image_ids) - self.n_cameras * self.sliding_window
                for _ in range(max(0, n_to_deregister)):
                    reg_image_ids = reconstruction.reg_image_ids()
                    reconstruction.deregister_image(image_id=min(reg_image_ids))

                # Scale the reconstruction (stereo rigs)
                if self.n_cameras != 1:
                    baselines = []
                    for img_id in list(set(self.keyframes_master_ids) & set(reconstruction.reg_image_ids())):
                        master = reconstruction.image(image_id=img_id)
                        timestamp = self.keyframes_ids[img_id].split("/")[1]
                        slave_id = self.keyframes_names.get(f"{self.second_baseline_camera}/{timestamp}")
                        if slave_id in reconstruction.reg_image_ids():
                            slave = reconstruction.image(image_id=slave_id)
                            baselines.append(np.linalg.norm(slave.projection_center() - master.projection_center()))
                    scale_factor = (np.median(baselines) / self.baseline) if baselines else 1.0
                else:
                    scale_factor = 1.0

                if self.test:
                    self.reconstruction_manager.write(self.out_dir)

            else:  # pycolmap reconstruct
                try:
                    self.controller.reconstruct(self.mapper_options)
                    reconstruction = self.reconstruction_manager.get(idx=0)
                except Exception:
                    self.reinitialize()
                    return [[image, None, None, None, None, None]], self.control_params, self.log_data

            if self.keyframe_count == self.config['mapping']['max_keyframes']:
                self.reconstruction_manager.write(self.out_dir)
                print("Max keyframes reached, exiting...")
                raise SystemExit

            if self.log:
                t1 = time.time()
                print(f"[CSLAM] Orientation time {t1 - t0:.2f} seconds")

            # --- Extract change in pose ---
            if self.log:
                t0 = time.time()
            try:
                if self.n_cameras == 1:
                    reconstruction = self.reconstruction_manager.get(idx=0)
                    last = reconstruction.image(image_id=self.keyframe_count - 1)
                    t = last.cam_from_world.translation
                    r = last.cam_from_world.rotation
                    reconstruction.transform(pycolmap.Sim3d({'translation': t, 'rotation': r, 'scale': 1}))

                    prev = reconstruction.image(image_id=self.keyframe_count - 2)
                    curr = reconstruction.image(image_id=self.keyframe_count)

                    if self.baseline_old == 0:
                        self.baseline_old = np.linalg.norm(curr.projection_center() - last.projection_center())
                        s = 1.0
                        delta_q = quat(curr.cam_from_world.rotation.quat)
                    else:
                        baseline = np.linalg.norm(last.projection_center() - prev.projection_center())
                        s = baseline / self.baseline_old if self.baseline_old != 0 else 1.0
                        self.baseline_old = np.linalg.norm(curr.projection_center() - last.projection_center())
                        delta_q = quat(curr.cam_from_world.rotation.quat)

                    delta_t = curr.projection_center() / s
                else:
                    reconstruction = self.reconstruction_manager.get(idx=0)
                    ref = reconstruction.image(image_id=self.keyframes_master_ids[-2])
                    t = ref.cam_from_world.translation
                    r = ref.cam_from_world.rotation
                    reconstruction.transform(pycolmap.Sim3d({'translation': t, 'rotation': r, 'scale': 1}))

                    curr = reconstruction.image(image_id=self.keyframes_master_ids[-1])
                    delta_q = quat(curr.cam_from_world.rotation.quat)

                    # scale_factor computed above (if custom path). If using pycolmap path and scale_factor undefined, default to 1.0
                    if 'scale_factor' not in locals():
                        scale_factor = 1.0
                    delta_t = curr.projection_center() / (scale_factor if scale_factor != 0 else 1.0)

                self.t_cumulative = self.t_cumulative + self.q_cumulative.inverse.rotate(delta_t)
                self.q_cumulative = delta_q * self.q_cumulative
            except Exception:
                if self.log:
                    print('[CSLAM] Error in estimate change pose')
                self.reinitialize()
                return [[image, None, None, None, None, None]], self.control_params, self.log_data

            if self.log:
                t1 = time.time()
                print(f"[CSLAM] Estimate change pose time {t1 - t0:.2f} seconds")
                print(f"[CSLAM] delta_t {delta_t}, delta_q [{delta_q}]")

            return [[image, self.keyframes_names[curr.name], delta_t, delta_q, self.t_cumulative, self.q_cumulative]], self.control_params, self.log

        # else path handled above when not promoted to keyframe
        return [[image, None, None, None, None, None]], self.control_params, self.log_data
