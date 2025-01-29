import os
import cv2
import time
import torch
import shutil
import numpy as np
import kornia.feature as KF

from pyquaternion import Quaternion
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from src.odometry.local_features import LocalFeatures
from src.odometry.db_colmap import COLMAPDatabase
from src.odometry.custom_incremental_pipeline import reconstruct

import pycolmap
from pycolmap import Database, Camera, Image, ListPoint2D, Rigid3d, Rotation3d, TwoViewGeometry, logging

def quat(colmap_quat: np.array) -> Quaternion:
    x = colmap_quat[0]
    y = colmap_quat[1]
    z = colmap_quat[2]
    w = colmap_quat[3]
    return Quaternion(np.array([w, x, y, z]))

class VisualOdometry:
    def __init__(
            self,
            working_dir: Path,
            config: dict,
            camera_config: dict,
    ) -> None:
        
        logging.verbose_level = 0
        logging.minloglevel = 2

        self.keyframes = []
        self.config = config
        self.camera_config = camera_config
        self.verbose = config['general']['verbose']
        self.height, self.width = camera_config['cam0']['height'], camera_config['cam0']['width']
        self.images_dir = working_dir / "images"
        self.test = self.config['general']['test']
        self.n_cameras = self.config['general']['n_cameras']
        self.keypoints, self.descriptors = {}, {}

        self.database_path = working_dir / "database.db"
        if self.database_path.exists():
            self.database_path.unlink()
        
        self.out_file_path = working_dir / "out.txt"
        if self.out_file_path.exists():
            self.out_file_path.unlink()
        
        self.out_dir = working_dir / "out"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.images = os.listdir(self.images_dir / "cam0")
        self.images.sort()

        self.local_features = LocalFeatures(
            self.width,
            self.height,
            config['local_features'],
            )
        if config['local_features']['features_name'] == "aliked":
            self.lightglue_model = "aliked"
        elif config['local_features']['features_name'] == "superpoint":
            self.lightglue_model = "superpoint"
        else:
            raise ValueError("Invalid local features model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lg_matcher = KF.LightGlueMatcher(self.lightglue_model).eval().to(self.device)

    def make_match_plot(
        self, img: np.ndarray, img2: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray, method: str = "flow",
    ) -> np.ndarray:
        if method == "flow":
            match_img = deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)
        elif method == "pair":
            img1_width = img.shape[1]
            img1_height = img.shape[0]
            match_img = np.concatenate((img, img2), axis=1)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0]+img1_width)), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(match_img, p1, 1, (0, 0, 255), 3, lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), 3, lineType=16)

        return match_img

    def match_features(self, keypoints, descriptors, pairs):
        matches = {}
        with torch.inference_mode():
            for pair in pairs:
                img1 = pair[0]
                img2 = pair[1]
                kps1, descs1 = keypoints[img1].to(self.device), descriptors[img1].to(self.device)
                kps2, descs2 = keypoints[img2].to(self.device), descriptors[img2].to(self.device)
                lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device))
                lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device))
                hw1 = np.array([self.height, self.width])
                hw2 = np.array([self.height, self.width])
                dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
                matches[(f'{img1}', f'{img2}')] = idxs
        return matches

    def match_distance(self, keypoints: dict, matches: dict, keyframe_name: str, frame_name: str) -> float:
        # Compute median match distance
        _matches = matches[(keyframe_name, frame_name)].cpu().numpy()
        mpts1 = keypoints[keyframe_name][_matches[:, 0]].cpu().numpy()
        mpts2 = keypoints[frame_name][_matches[:, 1]].cpu().numpy()
        match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
        median_match_dist = np.median(match_dist)

        ## Plot matches
        #plot = self.make_match_plot(cv2.imread(str(self.images_dir / keyframe_name)), cv2.imread(str(self.images_dir / frame_name)), mpts1, mpts2, method="pair")
        #plot_resized = cv2.resize(plot, (1920, 1080))
        #cv2.imshow("Image", plot_resized)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ##quit()

        return median_match_dist

    def write_keypoints_to_db(self, db: COLMAPDatabase, keyframe_name: str, image_id: int, camera_id: int, keypoints: dict) -> None:
        image = Image(
            name=keyframe_name,
            points2D=ListPoint2D(np.empty((0, 2), dtype=np.float64)),
            cam_from_world=Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0]),
            camera_id=camera_id,
            id=image_id,
            )
        db.write_image(image, use_image_id=True)
        db.write_keypoints(image_id=image_id, keypoints=keypoints[keyframe_name].cpu().numpy())

    def run(self) -> None:

        ## Visualize cam0 images
        #if self.test:
        #    for image_file in self.images:
        #        image_path = str(self.images_dir / 'cam0' / image_file)
        #        image = cv2.imread(image_path)
        #        cv2.imshow("Image", image)
        #        cv2.waitKey(1)
        #    cv2.destroyAllWindows()

        ## Rename Carla images
        #import os
        #for img in self.images:
        #    new_img = img[:-5] + ".jpg"
        #    os.rename(self.images_dir / 'cam0' / img, self.images_dir / 'cam0' / new_img)
        #for img in self.images:
        #    new_img = img[:-5] + ".jpg"
        #    os.rename(self.images_dir / 'cam1' / f"{img[:-4]}R.jpg", self.images_dir / 'cam1' / img)
        #quit()
        
        # Initialize database and odometry variables
        baseline = 0
        baseline_old = 0
        keyframe_count = 1
        keyframe_id = 1
        cumulative = np.array([0, 0, 0])
        cumulativa_quaternion = Quaternion(np.array([1, 0, 0, 0]))
        out_file = open(self.out_file_path, "a")

        db = Database(str(self.database_path))
        for c in range(self.n_cameras):
            camera = Camera(self.camera_config[f"cam{c}"])
            db.write_camera(camera)
            keyframe_name = f"cam{c}/{self.images[0]}"
            camera_id=1+c
            image_id=1+c
            new_keypoints, new_descriptors = self.local_features.extract(self.images_dir, image_files=[keyframe_name], batch_size=1)
            self.keypoints = self.keypoints | new_keypoints
            self.descriptors = self.descriptors | new_descriptors
            self.write_keypoints_to_db(db, keyframe_name, image_id, camera_id, self.keypoints)
        keyframe_name = f"cam0/{self.images[0]}"

        if self.n_cameras != 1:
            pairs = [(f"cam0/{self.images[0]}", f"cam1/{self.images[0]}")]
            matches = self.match_features(self.keypoints, self.descriptors, pairs)
            inlier_matches = matches[pairs[0]].cpu().numpy()
            db.write_two_view_geometry(
                1,
                2,
                TwoViewGeometry({"inlier_matches": inlier_matches})
                )

        # Mapper options
        sliding_window = self.config['mapping']['sliding_window']
        pycolmap.set_random_seed(0)
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.extract_colors = False
        options.fix_existing_images = False
        options.ba_global_max_num_iterations = 12 # Tested with 25 iterations
        options.ba_global_max_refinements = 1 # Tested with 5 refinements

        reconstruction_manager = pycolmap.ReconstructionManager()
        controller = pycolmap.IncrementalPipeline(
            options, str(self.images_dir), str(self.database_path), reconstruction_manager
        )

        mapper_options = controller.options.get_mapper()
        mapper_options.init_max_forward_motion = 0.99
        mapper_options.init_min_tri_angle = 1.0
        mapper_options.init_max_error = 100.0
        mapper_options.abs_pose_max_error = 50.0
        mapper_options.abs_pose_min_num_inliers = 8
        mapper_options.abs_pose_min_inlier_ratio = 0.05

        # Start odometry
        # Keyframe selection based on optical flow
        for frame_index in tqdm(range(1, len(self.images))):
            frame_name = f"cam0/{self.images[frame_index]}"
            new_keypoints, new_descriptors = self.local_features.extract(self.images_dir, image_files=[frame_name], batch_size=1)
            self.keypoints = self.keypoints | new_keypoints
            self.descriptors = self.descriptors | new_descriptors
            pairs = [(keyframe_name, frame_name)]
            matches = self.match_features(self.keypoints, self.descriptors, pairs)
            median_match_dist = self.match_distance(self.keypoints, matches, keyframe_name, frame_name)

            if median_match_dist < self.config['mapping']['max_match_distance']:
                del self.keypoints[frame_name]
                del self.descriptors[frame_name]
            
            # Matching on keyframes on master camera cam0
            if median_match_dist >= self.config['mapping']['max_match_distance']:
                inlier_matches = matches[(keyframe_name, frame_name)].cpu().numpy()
                keyframe_count += 1
                keyframe_id += 1*self.n_cameras
                keyframe_name = deepcopy(frame_name)
                camera_id = 1
                self.write_keypoints_to_db(db, keyframe_name, keyframe_id, camera_id, self.keypoints)
                db.write_two_view_geometry(
                    keyframe_id-1*self.n_cameras,
                    keyframe_id,
                    TwoViewGeometry({"inlier_matches": inlier_matches})
                    )
                
                if self.n_cameras != 1:
                    # Match slave cameras
                    slave_name = f"cam1/{self.images[frame_index]}"
                    slave_id = keyframe_id+1
                    camera_id = 2
                    new_keypoints, new_descriptors = self.local_features.extract(self.images_dir, image_files=[slave_name], batch_size=1)
                    self.keypoints = self.keypoints | new_keypoints
                    self.descriptors = self.descriptors | new_descriptors
                    self.write_keypoints_to_db(db, slave_name, slave_id, camera_id, self.keypoints)
                    pairs = [(keyframe_name, slave_name)]
                    matches = self.match_features(self.keypoints, self.descriptors, pairs)
                    inlier_matches = matches[(keyframe_name, slave_name)].cpu().numpy()
                    db.write_two_view_geometry(
                        keyframe_id,
                        slave_id,
                        TwoViewGeometry({"inlier_matches": inlier_matches})
                        )

                
                if keyframe_count > 5 and keyframe_count <= sliding_window:
                    controller.load_database()
                    if self.config['mapping']['method'] == 'custom':
                        reconstruct(controller, mapper_options, keyframe_id, False)
                    elif self.config['mapping']['method'] == 'with_pycolmap_reconstruct':
                        controller.reconstruct(mapper_options)
                    if self.test: reconstruction_manager.write(self.out_dir)

                if keyframe_count > sliding_window-1:
                    controller.load_database()
                    if self.test: reconstruction_manager.write(self.out_dir)
                    if self.config['mapping']['method'] == 'custom':
                        reconstruct(controller, mapper_options, keyframe_id, True)
                        if self.n_cameras == 2: reconstruct(controller, mapper_options, keyframe_id+1, True)
                        reconstruction = reconstruction_manager.get(idx=0)
                        reconstruction.deregister_image(image_id=keyframe_id-sliding_window*2+2)
                        if self.n_cameras == 2: reconstruction.deregister_image(image_id=keyframe_id+3-sliding_window*2)
                        if self.test: reconstruction_manager.write(self.out_dir)
                    elif self.config['mapping']['method'] == 'with_pycolmap_reconstruct':
                        controller.reconstruct(mapper_options)
                        reconstruction = reconstruction_manager.get(idx=0)
                        #reconstruction.deregister_image(image_id=keyframe_count+1-sliding_window)
                        #db.delete_inlier_matches(image_id1=keyframe_count+1-sliding_window, image_id2=keyframe_count-sliding_window) # ERROR ON INDEX OF 3D TIE POINTS (?) AFTER DELETING MATCHES IN DB
                    
                    if keyframe_count == self.config['mapping']['max_keyframes']:
                        reconstruction_manager.write(self.out_dir)
                        quit()

                    # Extract change in pose
                    if self.n_cameras == 1:
                        # Report the transformation on the lst_frm
                        lst_kfrm = reconstruction.image(image_id=keyframe_count)
                        t = lst_kfrm.cam_from_world.translation
                        r = lst_kfrm.cam_from_world.rotation
                        dict = {
                            'translation': t,
                            'rotation': r,
                            'scale': 1
                        }
                        reconstruction.transform(pycolmap.Sim3d(dict))

                        lst_lst_kfrm = reconstruction.image(image_id=keyframe_count-1)
                        lst_kfrm = reconstruction.image(image_id=keyframe_count)
                        new_kfrm = reconstruction.image(image_id=keyframe_count+1)

                        if baseline_old == 0:
                            baseline_old = np.linalg.norm(new_kfrm.projection_center() - lst_kfrm.projection_center())
                            s = 1
                            delta_q = quat(new_kfrm.cam_from_world.rotation.quat) # Output1
                        else:
                            baseline = np.linalg.norm(lst_kfrm.projection_center() - lst_lst_kfrm.projection_center())
                            s = baseline / baseline_old
                            baseline_old = np.linalg.norm(new_kfrm.projection_center() - lst_kfrm.projection_center())
                            delta_q = quat(new_kfrm.cam_from_world.rotation.quat) # Output1

                        delta_t = new_kfrm.projection_center()/s  # Output2
                        cumulative = deepcopy(cumulative) + cumulativa_quaternion.inverse.rotate(delta_t)
                        cumulativa_quaternion = delta_q * deepcopy(cumulativa_quaternion)
                        norm = cumulativa_quaternion.inverse.rotate(np.array([0, 0, 1]))
                        out_file.write(f"{cumulative[0]} {cumulative[1]} {cumulative[2]} {norm[0]} {norm[1]} {norm[2]}\n")
                        #out_file.write(f"{cumulative[0]}\t{cumulative[1]}\t{cumulative[2]}\t{baseline_old}\t{baseline}\t{np.linalg.norm(lst_kfrm.projection_center()/s - lst_lst_kfrm.projection_center()/s)}\n")
                    
                    elif self.n_cameras == 2:
                        # Report the transformation on the lst_frm
                        lst_lst_kfrm = reconstruction.image(image_id=keyframe_id-2)
                        t = lst_lst_kfrm.cam_from_world.translation
                        r = lst_lst_kfrm.cam_from_world.rotation
                        dict = {
                            'translation': t,
                            'rotation': r,
                            'scale': 1
                        }
                        reconstruction.transform(pycolmap.Sim3d(dict))

                        lst_lst_kfrm = reconstruction.image(image_id=keyframe_id-2)
                        lst_kfrm = reconstruction.image(image_id=keyframe_id)
                        new_kfrm = reconstruction.image(image_id=keyframe_id+1)
                        #print(new_kfrm.name, lst_kfrm.name, lst_lst_kfrm.name)

                        baseline = np.linalg.norm(new_kfrm.projection_center() - lst_kfrm.projection_center())
                        s = baseline/0.110078 # 1.5 CARLA  0.110078 EUROC
                        delta_q = quat(lst_kfrm.cam_from_world.rotation.quat) # Output1

                        delta_t = lst_kfrm.projection_center()/s  # Output2
                        cumulative = deepcopy(cumulative) + cumulativa_quaternion.inverse.rotate(delta_t)
                        cumulativa_quaternion = delta_q * deepcopy(cumulativa_quaternion)
                        norm = cumulativa_quaternion.inverse.rotate(np.array([0, 0, 1]))
                        out_file.write(f"{cumulative[0]} {cumulative[1]} {cumulative[2]} {norm[0]} {norm[1]} {norm[2]}\n")

        db.close()
        out_file.close()
        reconstruction_manager.write(self.out_dir)
        
        