import os
import cv2
import time
import copy
import torch
import shutil
import numpy as np
import kornia.feature as KF

from typing import List, Tuple, Dict, Any
from pyquaternion import Quaternion
from pathlib import Path

from src.odometry.local_features import LocalFeatures
from src.odometry.db_colmap import COLMAPDatabase
from src.odometry.custom_incremental_pipeline import reconstruct

import pycolmap
from pycolmap import Database, Camera, Image, Rigid3d, Rotation3d, TwoViewGeometry, logging
import traceback


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
    - comprehensive logging data collection
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
        self.current_status = 'initializing'
        
        # Enhanced logging data structure
        self.log_data: Dict[str, Any] = {
            'frame_count': 0,
            'keyframe_count': 0,
            'total_features_extracted': 0,
            'total_matches_computed': 0,
            'reinitializations': 0,
            'timing': {
                'feature_extraction_total': 0.0,
                'feature_matching_total': 0.0,
                'reconstruction_total': 0.0,
                'pose_estimation_total': 0.0,
                'frame_processing_total': 0.0,
                'database_operations_total': 0.0
            },
            'performance': {
                'avg_features_per_frame': 0.0,
                'avg_matches_per_pair': 0.0,
                'avg_frame_processing_time': 0.0,
                'keyframe_selection_ratio': 0.0
            },
            'reconstruction_stats': {
                'successful_reconstructions': 0,
                'failed_reconstructions': 0,
                'sliding_window_operations': 0,
                'bundle_adjustments': 0
            },
            'memory_stats': {
                'peak_gpu_memory_mb': 0.0,
                'current_cached_features': 0,
                'current_cached_lafs': 0
            },
            'current_frame': {
                'frame_name': '',
                'is_keyframe': False,
                'num_features': 0,
                'median_match_distance': 0.0,
                'processing_time': 0.0,
                'feature_extraction_time': 0.0,
                'matching_time': 0.0,
                'reconstruction_time': 0.0
            }
        }

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
        self.db = Database.open(str(self.database_path))
        self.db_dirty = False  # track when DB has new data not yet loaded into controller

        # Mapper options
        self.sliding_window = self.config['mapping']['sliding_window']
        pycolmap.set_random_seed(0)
        self.options = pycolmap.IncrementalPipelineOptions()
        self.options.ba_refine_focal_length = config['bundle_adjustment']['refine_focal_length']
        self.options.ba_refine_principal_point = config['bundle_adjustment']['refine_principal_point']
        self.options.ba_refine_extra_params = config['bundle_adjustment']['refine_extra_params']
        self.options.extract_colors = False
        self.options.fix_existing_frames = False
        self.options.ba_global_max_num_iterations = config['bundle_adjustment']['global_max_num_iterations']
        self.options.ba_global_max_refinements = config['bundle_adjustment']['global_max_refinements']
        self.options.multiple_models = False
        self.max_cost_change_px = config['bundle_adjustment']['max_cost_change_px']

        self.reconstruction_manager = pycolmap.ReconstructionManager()
        self.controller = pycolmap.IncrementalPipeline(
            self.options, str(self.images_dir), str(self.database_path), self.reconstruction_manager
        )

        self.mapper_options = self.controller.options.get_mapper()
        self.mapper_options.init_max_forward_motion = config['mapping']['init_max_forward_motion']
        self.mapper_options.init_min_tri_angle = config['mapping']['init_min_tri_angle']
        self.mapper_options.init_max_error = config['mapping']['init_max_error']
        self.mapper_options.abs_pose_max_error = config['mapping']['abs_pose_max_error']
        self.mapper_options.abs_pose_min_num_inliers = config['mapping']['abs_pose_min_num_inliers']
        self.mapper_options.abs_pose_min_inlier_ratio = config['mapping']['abs_pose_min_inlier_ratio']
        self.mapper_options.filter_max_reproj_error = config['mapping']['filter_max_reproj_error']
        self.mapper_options.abs_pose_max_error = 1000.0
        self.mapper_options.ba_local_min_tri_angle = 0.1
        self.controller.options.init_image_id1 = 3
        self.controller.options.init_image_id2 = 4
        #self.mapper_options.init_min_num_inliers = config['mapping']['init_min_num_inliers']
        #self.mapper_options.filter_min_tri_angle = config['mapping']['filter_min_tri_angle']
        #self.mapper_options.local_ba_min_tri_angle = config['mapping']['local_ba_min_tri_angle']
        #print(self.mapper_options);quit()

        # Precompute image size arrays (avoid realloc in matcher)
        self.hw_np = np.array([self.height, self.width])

        # for LightGlue: constant scale/orientation LAF base (ones) cached per image len
        self._ones_cache: Dict[int, torch.Tensor] = {}

    def _update_memory_stats(self):
        """Update memory statistics in log_data"""
        if torch.cuda.is_available():
            self.log_data['memory_stats']['peak_gpu_memory_mb'] = max(
                self.log_data['memory_stats']['peak_gpu_memory_mb'],
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
        
        self.log_data['memory_stats']['current_cached_features'] = len(self.keypoints)
        self.log_data['memory_stats']['current_cached_lafs'] = len(self.lafs_cache)

    def _update_performance_stats(self):
        """Update performance statistics in log_data"""
        frame_count = self.log_data['frame_count']
        keyframe_count = self.log_data['keyframe_count']
        
        if frame_count > 0:
            self.log_data['performance']['avg_frame_processing_time'] = (
                self.log_data['timing']['frame_processing_total'] / frame_count
            )
            self.log_data['performance']['keyframe_selection_ratio'] = keyframe_count / frame_count
            
        if self.log_data['total_features_extracted'] > 0 and frame_count > 0:
            self.log_data['performance']['avg_features_per_frame'] = (
                self.log_data['total_features_extracted'] / frame_count
            )

    def _log_timing(self, operation: str, duration: float):
        """Log timing for specific operations"""
        if operation in self.log_data['timing']:
            self.log_data['timing'][operation] += duration
        
        # Also log to current frame
        if operation == 'feature_extraction':
            if self.log_data['current_frame']['feature_extraction_time'] == 0:
                self.log_data['current_frame']['feature_extraction_time'] = duration
            else:
                self.log_data['current_frame']['feature_extraction_time'] += duration
        elif operation == 'feature_matching':
            self.log_data['current_frame']['matching_time'] = duration
        elif operation == 'reconstruction':
            self.log_data['current_frame']['reconstruction_time'] = duration

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
        total_matches = 0
        
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
            total_matches += len(idxs)
        
        # Update logging stats
        self.log_data['total_matches_computed'] += total_matches
        if len(pairs) > 0:
            self.log_data['performance']['avg_matches_per_pair'] = total_matches / len(pairs)
        
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
        
        # Log the median match distance
        self.log_data['current_frame']['median_match_distance'] = median_match_dist
        
        return float(median_match_dist)

    def write_keypoints_to_db(self, db: COLMAPDatabase, keyframe_name: str, image_id: int, camera_id: int, keypoints: Dict[str, torch.Tensor]) -> None:
        db_start = time.time()
        
        image = Image(
            name=keyframe_name,
            points2D=np.empty((0, 2), dtype=np.float64),
            camera_id=camera_id,
            #id=image_id,
        )
        rigid = Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0])
        #pose_matrix = rigid.matrix()  # 4x4 transformation
        #pose = Pose.from_cam_to_world(pose_matrix)
        #cam_from_world=Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0])
        #image.pose = rigid
        db.write_image(image, use_image_id=False)
        db.write_keypoints(image_id=image_id, keypoints=keypoints[keyframe_name].detach().cpu().numpy())
        self.db_dirty = True
        
        db_time = time.time() - db_start
        self._log_timing('database_operations_total', db_time)

    def _extract_and_store(self, name: str, img: np.ndarray) -> int:
        """Extract and store features, return number of features extracted"""
        extraction_start = time.time()
        
        new_kps, new_descs, reading_images_status = self.local_features.extract(name, img)
        if not reading_images_status:
            return 0, reading_images_status
        
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
        
        extraction_time = time.time() - extraction_start
        self._log_timing('feature_extraction_total', extraction_time)
        self._log_timing('feature_extraction', extraction_time)
        
        # Count features and update stats
        num_features = sum(len(kps) for kps in new_kps.values())
        self.log_data['total_features_extracted'] += num_features
        self.log_data['current_frame']['num_features'] = num_features
        
        return num_features, reading_images_status

    def reinitialize(self) -> None:
        self.current_status = 'initializing'
        if self.log:
            print('[CSLAM:] Reinitializing..')
        
        # Update reinitalization count
        self.log_data['reinitializations'] += 1
        self.N_reinit += 1
        
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
            db_start = time.time()
            self.controller.load_database()
            self.db_dirty = False
            db_time = time.time() - db_start
            self._log_timing('database_operations_total', db_time)

    def run(
        self, image: str,
        images: List[np.ndarray],
        reinitialize: bool,
    ):
        frame_start_time = time.time()
        
        # Initialize current frame data
        self.log_data['current_frame'] = {
            'status': self.current_status,
            'frame_name': image,
            'is_keyframe': False,
            'num_features': 0,
            'median_match_distance': 0.0,
            'processing_time': 0.0,
            'feature_extraction_time': 0.0,
            'matching_time': 0.0,
            'reconstruction_time': 0.0,
            'corrupted_master_image': False,
            'corrupted_slave_image': False,
            'not_enough_features_on_master': False,
        }
        
        self.images.append(image)
        self.log_data['frame_count'] += 1

        if reinitialize:
            self.reinitialize()
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            return [[image, None, None, None, None, None]], self.log_data

        if len(self.keyframes_names.keys()) == 0:
            # first batch: register all cameras for the first timestamp
            status = []
            rig_num_features = []
            rig = {}
            for c, cam in enumerate(self.cameras):
                camera = Camera(self.camera_config[f"{cam}"])
                self.db.write_camera(camera)
                keyframe_name = f"{cam}/{image}"
                camera_id = 1 + c
                image_id = 1 + c
                num_features, reading_images_status = self._extract_and_store(keyframe_name, images[c])
                status.append(reading_images_status)
                rig_num_features.append(num_features)
                rig[c] = (keyframe_name, image_id, camera_id)

            # Check if all images were read successfully and have features
            if not all(status):
                self.log_data['current_frame']['corrupted_master_image'] = True
                return [[image, None, None, None, None, None]], self.log_data
            elif not all(rig_num_features):
                self.log_data['current_frame']['not_enough_features_on_master'] = True
                return [[image, None, None, None, None, None]], self.log_data 
            else:
                for c, cam in enumerate(self.cameras):
                    keyframe_name = rig[c][0]
                    image_id = rig[c][1]
                    camera_id = rig[c][2]
                    self.write_keypoints_to_db(self.db, keyframe_name, image_id, camera_id, self.keypoints)
                    self.keyframes_names[keyframe_name] = image_id
                    self.keyframes_ids[image_id] = keyframe_name

            self.keyframe_name = f"cam0/{image}"
            self.keyframes_master_ids.append(1)
            self.log_data['current_frame']['is_keyframe'] = True
            self.log_data['keyframe_count'] += 1

            if self.n_cameras != 1:
                matching_start = time.time()
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
                matching_time = time.time() - matching_start
                self._log_timing('feature_matching_total', matching_time)
                self._log_timing('feature_matching', matching_time)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            
            return [[image, None, None, None, None, None]], self.log_data

        # -------- Keyframe selection based on feature motion --------
        matching_start = time.time()
        frame_name = f"cam0/{image}"
        num_features, reading_images_status = self._extract_and_store(frame_name, images[0])
        if not reading_images_status:
                self.log_data['current_frame']['corrupted_master_image'] = True
                return [[image, None, None, None, None, None]], self.log_data
        elif num_features==0:
            self.log_data['current_frame']['not_enough_features_on_master'] = True
            return [[image, None, None, None, None, None]], self.log_data 

        pairs = [(self.keyframe_name, frame_name)]
        matches = self.match_features(self.keypoints, self.descriptors, pairs)
        median_match_dist = self.match_distance(self.keypoints, matches, self.keyframe_name, frame_name)
        matching_time = time.time() - matching_start
        self._log_timing('feature_matching_total', matching_time)
        self._log_timing('feature_matching', matching_time)

        if median_match_dist < self.config['mapping']['max_match_distance']:
            # not a keyframe: free memory of temporary frame
            self.keypoints.pop(frame_name, None)
            self.descriptors.pop(frame_name, None)
            self.lafs_cache.pop(frame_name, None)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self.log_data['current_frame']['status'] = self.current_status
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            
            return [[image, None, None, None, None, None]], self.log_data

        # --- Promote to keyframe on master camera ---
        self.log_data['current_frame']['is_keyframe'] = True
        self.log_data['keyframe_count'] += 1
        
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
            slave_matching_start = time.time()
            for c, cam in enumerate(self.cameras):
                if c == 0:
                    continue
                slave_name = f"{cam}/{image}"
                slave_id = self.keyframe_id + 1 * c
                self.keyframes_names[slave_name] = slave_id
                self.keyframes_ids[slave_id] = slave_name
                camera_id = c + 1
                num_features, reading_images_status = self._extract_and_store(slave_name, images[c])
                if not reading_images_status:
                    self.keypoints[slave_name] = np.empty((0, 2), dtype=np.float32)
                    self.descriptors[slave_name] = np.empty((0, self.local_features.decriptor_dim), dtype=np.float32)
                    self.log_data['current_frame']['corrupted_slave_image'] = True
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
            slave_matching_time = time.time() - slave_matching_start
            self._log_timing('feature_matching_total', slave_matching_time)

        if self.log:
            print(f"[CSLAM] Matching time: {matching_time:.2f} seconds")

        # --- Orientation / Reconstruction step ---
        reconstruction_start = time.time()

        if 5 < self.keyframe_count < self.sliding_window:
            self.current_status = 'reconstruction_initialization'
            self._maybe_load_db()
            try:
                if self.config['mapping']['method'] == 'custom':
                    cucu = reconstruct(self.controller, self.mapper_options, self.keyframe_id, False, run_BA=True, max_cost_change_px=self.max_cost_change_px)
                    self.log_data['reconstruction_stats']['bundle_adjustments'] += 1
                    print("CIAO")
                    print(cucu);quit()
                else:
                    self.controller.reconstruct(self.mapper_options)
                self.log_data['reconstruction_stats']['successful_reconstructions'] += 1
            except Exception as e:
                self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                self.reinitialize()
                frame_time = time.time() - frame_start_time
                self.log_data['current_frame']['processing_time'] = frame_time
                if self.log:
                    print(f"[CSLAM] Error in reconstruction: {str(e)}")
                    traceback.print_exc()
                return [[image, None, None, None, None, None]], self.log_data
            if self.test:
                self.reconstruction_manager.write(self.out_dir)
            
            reconstruction_time = time.time() - reconstruction_start
            self._log_timing('reconstruction_total', reconstruction_time)
            self._log_timing('reconstruction', reconstruction_time)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            
            return [[image, None, None, None, None, None]], self.log_data

        elif self.keyframe_count >= self.sliding_window:
            self.current_status = 'orientation'
            self._maybe_load_db()

            if self.config['mapping']['method'] == 'custom':
                # Add new keyframes (entire rig)
                for c in range(self.n_cameras):
                    try:
                        if c == self.n_cameras - 1:
                            self.run_BA = True
                        if self.log:
                            tt0 = time.time()
                        reconstruct(self.controller, self.mapper_options, self.keyframe_id + c, True, run_BA=self.run_BA, max_cost_change_px=self.max_cost_change_px)
                        if self.run_BA:
                            self.log_data['reconstruction_stats']['bundle_adjustments'] += 1
                        if self.log:
                            tt1 = time.time()
                            print('[CSLAM] Time for reconstruct:', tt1 - tt0)
                        self.run_BA = False
                    except Exception as e:
                        if self.log:
                            print('[CSLAM] Error in keyframe orientation:', str(e))
                            traceback.print_exc()
                        self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                        self.reinitialize()
                        frame_time = time.time() - frame_start_time
                        self.log_data['current_frame']['processing_time'] = frame_time
                        return [[image, None, None, None, None, None]], self.log_data

                # Sliding window: deregister oldest
                reconstruction = self.reconstruction_manager.get(idx=0)
                reg_image_ids = reconstruction.reg_image_ids()
                n_to_deregister = len(reg_image_ids) - self.n_cameras * self.sliding_window
                for _ in range(max(0, n_to_deregister)):
                    reg_image_ids = reconstruction.reg_image_ids()
                    reconstruction.deregister_image(image_id=min(reg_image_ids))
                    self.log_data['reconstruction_stats']['sliding_window_operations'] += 1

                self.log_data['reconstruction_stats']['successful_reconstructions'] += 1

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
                    self.log_data['reconstruction_stats']['successful_reconstructions'] += 1
                except Exception as e:
                    self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                    self.reinitialize()
                    frame_time = time.time() - frame_start_time
                    self.log_data['current_frame']['processing_time'] = frame_time
                    if self.log:
                        print(f"[CSLAM] Error in reconstruction: {str(e)}")
                        traceback.print_exc()
                    return [[image, None, None, None, None, None]], self.log_data

            reconstruction_time = time.time() - reconstruction_start
            self._log_timing('reconstruction_total', reconstruction_time)
            self._log_timing('reconstruction', reconstruction_time)

            if self.keyframe_count == self.config['mapping']['max_keyframes']:
                self.reconstruction_manager.write(self.out_dir)
                print("Max keyframes reached, exiting...")
                raise SystemExit

            if self.log:
                print(f"[CSLAM] Orientation time {reconstruction_time:.2f} seconds")

            # --- Extract change in pose ---
            pose_start = time.time()
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
                    print('************************************', t, r)
                    cov = pycolmap.BACovariance()
                    aaa = cov.get_cam_cov_from_world(image_id=self.keyframes_master_ids[-2])
                    print('************************************', aaa)
                    reconstruction.transform(pycolmap.Sim3d({'translation': t, 'rotation': r, 'scale': 1}))

                    curr = reconstruction.image(image_id=self.keyframes_master_ids[-1])
                    delta_q = quat(curr.cam_from_world.rotation.quat)

                    # scale_factor computed above (if custom path). If using pycolmap path and scale_factor undefined, default to 1.0
                    if 'scale_factor' not in locals():
                        scale_factor = 1.0
                    delta_t = curr.projection_center() / (scale_factor if scale_factor != 0 else 1.0)

                self.t_cumulative = self.t_cumulative + self.q_cumulative.inverse.rotate(delta_t)
                self.q_cumulative = delta_q * self.q_cumulative
                
                pose_time = time.time() - pose_start
                self._log_timing('pose_estimation_total', pose_time)
                
            except Exception as e:
                if self.log:
                    print('[CSLAM] Error in estimate change pose:', str(e))
                    traceback.print_exc()
                self.reinitialize()
                frame_time = time.time() - frame_start_time
                self.log_data['current_frame']['processing_time'] = frame_time
                quit()
                return [[image, None, None, None, None, None]], self.log_data

            if self.log:
                print(f"[CSLAM] Estimate change pose time {pose_time:.2f} seconds")
                print(f"[CSLAM] delta_t {delta_t}, delta_q [{delta_q}]")

            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()

            return [[image, self.keyframes_names[curr.name], delta_t, delta_q, self.t_cumulative, self.q_cumulative]], self.log_data

        # else path handled above when not promoted to keyframe
        frame_time = time.time() - frame_start_time
        self.log_data['current_frame']['processing_time'] = frame_time
        self._log_timing('frame_processing_total', frame_time)
        self._update_memory_stats()
        self._update_performance_stats()
        
        return [[image, None, None, None, None, None]], self.log_data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary"""
        return {
            'frame_statistics': {
                'total_frames': self.log_data['frame_count'],
                'total_keyframes': self.log_data['keyframe_count'],
                'keyframe_ratio': self.log_data['performance']['keyframe_selection_ratio'],
                'total_reinitializations': self.log_data['reinitializations']
            },
            'timing_summary': {
                'total_processing_time': self.log_data['timing']['frame_processing_total'],
                'avg_frame_time': self.log_data['performance']['avg_frame_processing_time'],
                'feature_extraction_total': self.log_data['timing']['feature_extraction_total'],
                'feature_matching_total': self.log_data['timing']['feature_matching_total'],
                'reconstruction_total': self.log_data['timing']['reconstruction_total'],
                'pose_estimation_total': self.log_data['timing']['pose_estimation_total']
            },
            'feature_statistics': {
                'total_features_extracted': self.log_data['total_features_extracted'],
                'avg_features_per_frame': self.log_data['performance']['avg_features_per_frame'],
                'total_matches_computed': self.log_data['total_matches_computed']
            },
            'reconstruction_statistics': self.log_data['reconstruction_stats'],
            'memory_statistics': self.log_data['memory_stats']
        }