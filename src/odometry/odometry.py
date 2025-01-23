import os
import cv2
import time
import torch
import pycolmap
import numpy as np
import kornia.feature as KF

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from pycolmap import Database, Camera, Image, ListPoint2D, Rigid3d, Rotation3d, TwoViewGeometry
from src.odometry.local_features import LocalFeatures
from src.odometry.db_colmap import COLMAPDatabase
from src.odometry.custom_incremental_pipeline import reconstruct


class VisualOdometry:
    def __init__(
            self,
            working_dir: Path,
            config: dict,
            camera_config: dict,
    ) -> None:
        self.height, self.width = camera_config['height'], camera_config['width']
        self.database_path = working_dir / "database.db"
        self.images_dir = working_dir / "images"
        self.config = config
        self.camera_config = camera_config
        self.test = self.config['general']['test']
        self.keyframes = []
        self.images = os.listdir(self.images_dir)
        self.images.sort()
        self.local_features = LocalFeatures(
            self.camera_config['width'],
            self.camera_config['height'],
            config['local_features'],
            )
        if config['local_features']['features_name'] == "aliked":
            self.lightglue_model = "aliked"
        elif config['local_features']['features_name'] == "superpoint":
            self.lightglue_model = "superpoint"
        else:
            raise ValueError("Invalid local features model")

    def make_match_plot(
        self, img: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray
    ) -> np.ndarray:
        match_img = deepcopy(img)
        for pt1, pt2 in zip(mpts1, mpts2):
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
            cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)

        return match_img

    def match_features(self, keypoints, descriptors, pairs):
        matches = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lg_matcher = KF.LightGlueMatcher(self.lightglue_model).eval().to(device)
        with torch.inference_mode():
            for pair in tqdm(pairs):
                img1 = pair[0]
                img2 = pair[1]
                kps1, descs1 = keypoints[img1], descriptors[img1]
                kps2, descs2 = keypoints[img2], descriptors[img2]
                lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
                hw1 = np.array([self.height, self.width])
                hw2 = np.array([self.height, self.width])
                dists, idxs = lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
                matches[(f'{img1}', f'{img2}')] = idxs
        return matches

    def run(self) -> None:
        #if self.test:
        #    for image_file in self.images:
        #        image_path = str(self.images_dir / image_file)
        #        image = cv2.imread(image_path)
        #        cv2.imshow("Image", image)
        #        cv2.waitKey(1)
        #    cv2.destroyAllWindows()
        
        # Initialize database
        baseline = 0
        baseline_old = 0
        cumulative = np.array([0, 0, 0])
        out_file_path = Path("./work_dir/out.txt")
        if out_file_path.exists():
            out_file_path.unlink()
        out_file = open(out_file_path, "a")
        keyframe_count = 0
        keyframe_name = self.images[0]
        keypoints, descriptors = self.local_features.extract(self.images_dir, image_files=[keyframe_name], batch_size=1)

        if self.database_path.exists():
            self.database_path.unlink()
            print("Database deleted")
        start = time.time()
        db = Database(str(self.database_path))
        camera = Camera(self.camera_config)
        db.write_camera(camera)
        image0 = Image(
            name=keyframe_name,
            points2D=ListPoint2D(np.empty((0, 2), dtype=np.float64)),
            cam_from_world=Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0]),
            camera_id=1,
            id=1,
            )
        db.write_image(image0, use_image_id=True)
        db.write_keypoints(image_id=1, keypoints=keypoints[keyframe_name].cpu().numpy())
        
        end = time.time()
        print(f"Time: {end-start}")

        # Start odometry
        sliding_window = 50 # 30
        pycolmap.set_random_seed(0)
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.extract_colors = False
        options.fix_existing_images = False # False or True
        options.ba_global_max_num_iterations = 25 # 25
        #print(options);quit()
        reconstruction_manager = pycolmap.ReconstructionManager()
        controller = pycolmap.IncrementalPipeline(
            options, str(self.images_dir), str(self.database_path), reconstruction_manager
        )

        mapper_options = controller.options.get_mapper()
        mapper_options.init_max_forward_motion = 0.99
        mapper_options.init_min_tri_angle = 1.0
        for frame_index in range(1, len(self.images)):
            frame_name = self.images[frame_index]
            new_keypoints, new_descriptors = self.local_features.extract(self.images_dir, image_files=[frame_name], batch_size=1)
            keypoints = keypoints | new_keypoints
            descriptors = descriptors | new_descriptors
            pairs = [(keyframe_name, frame_name)]
            matches = self.match_features(keypoints, descriptors, pairs)
            matches = matches[(keyframe_name, frame_name)].cpu().numpy()
            mpts1 = keypoints[keyframe_name][matches[:, 0]].cpu().numpy()
            mpts2 = keypoints[frame_name][matches[:, 1]].cpu().numpy()
            match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
            median_match_dist = np.median(match_dist)
            print(median_match_dist)
            #plot = self.make_match_plot(cv2.imread(str(self.images_dir / keyframe_name)), mpts1, mpts2)
            #cv2.imshow("Image", plot)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #if frame_index == 100:
            #    quit()
            
            if median_match_dist > 10:
                keyframe_count += 1
                keyframe_name = deepcopy(frame_name)
                # Elimiare i keypoints dei frames che non sono keyframes
                image = Image(
                    name=keyframe_name,
                    points2D=ListPoint2D(np.empty((0, 2), dtype=np.float64)),
                    cam_from_world=Rigid3d(rotation=Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0]),
                    camera_id=1,
                    id=1+keyframe_count,
                    )
                db.write_image(image, use_image_id=True)
                db.write_keypoints(image_id=1+keyframe_count, keypoints=keypoints[keyframe_name].cpu().numpy())
                #db.write_matches()
                two_view_geom = TwoViewGeometry({"inlier_matches": matches})
                print(two_view_geom.todict())
                db.write_two_view_geometry(keyframe_count, 1+keyframe_count, two_view_geom)

                if keyframe_count > 5 and keyframe_count <= sliding_window:
                    controller.load_database()
                    if self.config['mapping']['method'] == 'custom':
                        reconstruct(controller, mapper_options, 1+keyframe_count, False)
                    elif self.config['mapping']['method'] == 'with_pycolmap_reconstruct':
                        controller.reconstruct(mapper_options)
                    if self.test: reconstruction_manager.write("./work_dir/out")

                if keyframe_count > sliding_window-1:
                    controller.load_database()
                    if self.test: reconstruction_manager.write("./work_dir/out")
                    if self.config['mapping']['method'] == 'custom':
                        reconstruct(controller, mapper_options, 1+keyframe_count, True)
                        reconstruction = reconstruction_manager.get(idx=0)
                        reconstruction.deregister_image(image_id=keyframe_count+1-sliding_window)
                    elif self.config['mapping']['method'] == 'with_pycolmap_reconstruct':
                        controller.reconstruct(mapper_options)
                        reconstruction = reconstruction_manager.get(idx=0)
                        #reconstruction.deregister_image(image_id=keyframe_count+1-sliding_window)
                        #db.delete_inlier_matches(image_id1=keyframe_count+1-sliding_window, image_id2=keyframe_count-sliding_window) # ERROR ON INDEX OF 3D TIE POINTS (?) AFTER DELETING MATCHES IN DB
                    
                    #if reconstruction.num_reg_images() > 15:
                    if self.test: reconstruction_manager.write("./work_dir/out1")
                    if keyframe_count == self.config['mapping']['max_keyframes']:
                        reconstruction_manager.write("./work_dir/out1")
                        quit()
                    
                    # If I rememeber well, did not work this approach. So I used reconstruction.deregister_image(image_id=keyframe_count+1-sliding_window)
                    #to_deregister_image = reconstruction.image(image_id=keyframe_count+1-10)
                    #if to_deregister_image.has_pose:
                    #    to_deregister_image.reset_pose()

                    # Report the transformation on the lst_frm
                    for i in reconstruction.reg_image_ids():
                        image = reconstruction.image(image_id=i)
                        print(image.projection_center())
                    lst_kfrm = reconstruction.image(image_id=keyframe_count)
                    t = lst_kfrm.projection_center() 
                    r = lst_kfrm.cam_from_world.rotation
                    #transformation = lst_kfrm.cam_from_world
                    #transformation_inverse = transformation.inverse().todict()
                    #transformation_inverse['scale'] = 1
                    #reconstruction.transform(transformation_inverse)
                    print('t:', -t)
                    dict = {
                        'translation': -t,
                        'rotation': r,
                        'scale': 1
                    }
                    reconstruction.transform(pycolmap.Sim3d(dict))
                    for i in reconstruction.reg_image_ids():
                        image = reconstruction.image(image_id=i)
                        print(image.projection_center())


                    # Extract info 3d
                    lst_lst_kfrm = reconstruction.image(image_id=keyframe_count-1)
                    lst_kfrm = reconstruction.image(image_id=keyframe_count)
                    #try:
                    new_kfrm = reconstruction.image(image_id=keyframe_count+1)
                    #except:
                    #    reconstruction_manager.write("./work_dir/out")
                    if baseline_old == 0:
                        baseline_old = np.linalg.norm(new_kfrm.projection_center() - lst_kfrm.projection_center())
                        s = 1
                    else:
                        baseline = np.linalg.norm(lst_kfrm.projection_center() - lst_lst_kfrm.projection_center())
                        s = baseline / baseline_old
                        baseline_old = np.linalg.norm(new_kfrm.projection_center() - lst_kfrm.projection_center())
                    delta = (new_kfrm.projection_center() - lst_kfrm.projection_center())/s
                    cumulative = deepcopy(cumulative) + deepcopy(delta)
                    print(cumulative)
                    out_file.write(f"{cumulative[0]}\t{cumulative[1]}\t{cumulative[2]}\t{baseline_old}\t{baseline}\t{np.linalg.norm(lst_kfrm.projection_center()/s - lst_lst_kfrm.projection_center()/s)}\n")
                    #db.
                    #reconstruction_manager.write("./work_dir/out")
                    


        db.close()
        out_file.close()
        reconstruction_manager.write("./work_dir/out")
        
        