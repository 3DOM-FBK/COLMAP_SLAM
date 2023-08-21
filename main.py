import logging
import os
import pickle
import shutil
import subprocess
import time
import glob

import cv2
import numpy as np
import pydegensac
from pyquaternion import Quaternion

from lib import (
    ExtractCustomFeatures,
    database,
    db_colmap,
    export_cameras,
    import_local_features,
    matcher,
    utils,
)
from lib.colmapAPI import ColmapAPI
from lib.keyframe_selection import KeyFrameSelConfFile, KeyFrameSelector
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatConfFile, LocalFeatureExtractor
from lib import cameras
from pathlib import Path

# Configuration file
CFG_FILE = "config.ini"

# Setup logging level
LOG_LEVEL = logging.INFO
utils.Inizialization.setup_logger(LOG_LEVEL)
logger = logging.getLogger("ColmapSLAM")

# Inizialize COLMAP SLAM problem
init = utils.Inizialization(CFG_FILE)
cfg = init.inizialize()
cfg.PLOT_TRJECTORY = False # Please keep PLOT_TRAJECTORY = False.
# It is used to plot an additional trajectory (matplotlib) but for now shows rarely a bug (multiple plots).
# In any case the plot from opencv will show

# Initialize variables
SEQUENTIAL_OVERLAP = cfg.SEQUENTIAL_OVERLAP
MAX_SEQUENTIAL_OVERLAP = 50
if cfg.SNAPSHOT == True:
    SNAPSHOT_DIR = './frames'
elif cfg.SNAPSHOT == False:
    SNAPSHOT_DIR = None
keyframes_list = KeyFrameList()
processed_imgs = []
kfm_batch = []
pointer = 0  # pointer points to the last oriented image
delta = 0  # delta is equal to the number of processed but not oriented imgs
first_colmap_loop = True
one_time = False  # It becomes true after the first batch of images is oriented
# The first batch of images define the reference system.
# At following epochs the photogrammetric model will be reported in this ref system.
reference_imgs = []
adjacency_matrix = None

# Setup keyframe selector
kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
keyframe_selector = KeyFrameSelector(
    keyframes_list=keyframes_list,
    last_keyframe_pointer=pointer,
    last_keyframe_delta=delta,
    keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
    kfs_method=cfg.KFS_METHOD,
    geometric_verification="pydegensac",
    local_feature=cfg.KFS_LOCAL_FEATURE,
    local_feature_cfg=kf_selection_detecor_config,
    n_features=cfg.KFS_N_FEATURES,
    realtime_viz=True,
    viz_res_path=SNAPSHOT_DIR,
    innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
    min_matches=cfg.MIN_MATCHES,
    error_threshold=cfg.RANSAC_THRESHOLD,
    iterations=cfg.RANSAC_ITERATIONS,
    n_camera=cfg.N_CAMERAS,
)

# Setup local feature to use on keyframes
local_feat_conf = LocalFeatConfFile(cfg)
local_feat_extractor = LocalFeatureExtractor(
    cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
)

# If the camera coordinates are known from other sensors than gnss,
# they can be stores in camera_coord_other_sensors dictionary and used
# to scale the photogrammetric model
camera_coord_other_sensors = {}
if cfg.USE_EXTERNAL_CAM_COORD == True:
    with open(cfg.CAMERA_COORDINATES_FILE, "r") as gt_file:
        lines = gt_file.readlines()
        for line in lines[2:]:
            id, x, y, z, _ = line.split(" ", 4)
            camera_coord_other_sensors[id] = (x, y, z)

# Stream of input data
if cfg.USE_SERVER == True:
    stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
else:
    stream_proc = subprocess.Popen(["python", "./simulator.py"])

# Set-up plotqq
# create_plot()
if cfg.PLOT_TRJECTORY:
    plot_proc = subprocess.Popen(["python", "./plot.py"])

# Initialize COLMAP API
colmap = ColmapAPI(str(cfg.COLMAP_EXE_PATH))


# MAIN LOOP
timer_global = utils.AverageTimer(logger=logger)
while True:
    # Get sorted image list available in imgs folders
    imgs = sorted((cfg.IMGS_FROM_SERVER / "cam0").glob(f"*.{cfg.IMG_FORMAT}"))

    ## If using the simulator, check if process is still alive, otherwise quit
    #if cfg.USE_SERVER == False:
    #    if stream_proc.poll() is not None:
    #        logging.info("Simulator completed.")
    #        if cfg.PLOT_TRJECTORY:
    #            plot_proc.kill()
    #        break
    #else:
    #    # Make exit condition when using server
    #    pass

    newer_imgs = False  # To control that new keyframes are added
    processed = 0  # Number of processed images

    # Keyframe selection
    if len(imgs) < 1:
        continue

    if len(imgs) < 2:
        # Set first frame as keyframe
        img0 = imgs[pointer]
        existing_keyframe_number = 0
        for c in range(cfg.N_CAMERAS):
            shutil.copy(
                img0.parent.parent / f"cam{c}" / img0.name,
                cfg.KF_DIR_BATCH / f"cam{c}" / f"{utils.Id2name(existing_keyframe_number)}",
            )
        camera_id = 1
        if img0 not in keyframes_list.keyframes_names:
            keyframes_list.add_keyframe(
                KeyFrame(
                    img0,
                    existing_keyframe_number,
                    utils.Id2name(existing_keyframe_number),
                    camera_id,
                    pointer + delta + 1,
                )
            )
            continue

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            # Only new images found in the target folder are processed.
            # No more than MAX_IMG_BATCH_SIZE imgs are processed.
            if img not in processed_imgs and c == 0:
                processed_imgs.append(img)
                processed += 1
                continue
            if img in processed_imgs or processed >= cfg.MAX_IMG_BATCH_SIZE:
                continue

            img1 = imgs[pointer]
            img2 = img

            logger.info(f"\nProcessing image pair ({img1}, {img2})")
            logger.info(f"pointer {pointer} c {c}")

            old_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))

            (
                keyframes_list,
                pointer,
                delta,
                kfs_time,
            ) = keyframe_selector.run(img1, img2)

            # Set if new keyframes are added
            new_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
            if new_n_keyframes - old_n_keyframes > 0:
                newer_imgs = True
                kfm_batch.append(img.name)
                keyframe_obj = keyframes_list.get_keyframe_by_image_name(img)
                with open('keyframes.txt', 'a') as kfm_imgs:
                    kfm_imgs.write(f"{keyframe_obj._image_name},{cfg.KF_DIR_BATCH}/cam0/{keyframe_obj._keyframe_name}\n")

                ## Load exif data and store GNSS position if present
                ## or load camera cooridnates from other sensors
                ## Exif gnss data works properly only when directions are N and E
                # exif_data = []
                # try:
                #    exif_data = piexif.load(str(img2))
                # except:
                #    logger.error(
                #        "Error loading exif data. Image file could be corrupted."
                #    )
            #
            # if exif_data != [] and len(exif_data["GPS"].keys()) != 0:
            #    lat = exif_data["GPS"][2]
            #    long = exif_data["GPS"][4]
            #    alt = exif_data["GPS"][6]
            #    enuX, enuY, enuZ = ConvertGnssRefSystm.CovertGnssRefSystm(
            #        lat, long, alt
            #    )
            #    keyframe_obj.GPSLatitude = lat
            #    keyframe_obj.GPSLongitude = long
            #    keyframe_obj.GPSAltitude = alt
            #    keyframe_obj.enuX = enuX
            #    keyframe_obj.enuY = enuY
            #    keyframe_obj.enuZ = enuZ
            #
            # elif exif_data != [] and img2 in camera_coord_other_sensors.keys():
            #    print("img2", img2)
            #    enuX, enuY, enuZ = (
            #        camera_coord_other_sensors[img2][0],
            #        camera_coord_other_sensors[img2][1],
            #        camera_coord_other_sensors[img2][2],
            #    )
            #    keyframe_obj.enuX = enuX
            #    keyframe_obj.enuY = enuY
            #    keyframe_obj.enuZ = enuZ

            processed_imgs.append(img)
            processed += 1

    # INCREMENTAL RECONSTRUCTION
    kfrms = os.listdir(cfg.KF_DIR_BATCH / "cam0")
    kfrms.sort()    

    if len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs == True:
        timer = utils.AverageTimer(logger=logger)
        logger.info(f"DYNAMIC MATCHING WINDOW: {cfg.SEQUENTIAL_OVERLAP}")

        if first_colmap_loop == True:
            logger.info("Initialize an empty database")
            colmap.CreateEmptyDatabase(cfg.DATABASE)
            timer.update("DATABASE INITIALIZATION")

        logger.info("Feature extraction")
        if cfg.LOCAL_FEAT_LOCAL_FEATURE != "RootSIFT":
            local_feat_extractor.run(cfg.DATABASE, cfg.KF_DIR_BATCH, cfg.IMG_FORMAT)
        elif cfg.LOCAL_FEAT_LOCAL_FEATURE == "RootSIFT":
            colmap.ExtractRootSiftFeatures(
                database_path=cfg.DATABASE,
                path_to_images=cfg.KF_DIR_BATCH,
                first_loop=first_colmap_loop,
                max_n_features=cfg.LOCAL_FEAT_N_FEATURES,
            )
        
        cameras.AssignCameras(cfg.DATABASE, len(cfg.CAM))

        timer.update("FEATURE EXTRACTION")

        logger.info("Sequential matcher")

        if cfg.LOOP_CLOSURE_DETECTION == False:
            if first_colmap_loop == True:
                old_adjacency_matrix_shape = 0
            elif first_colmap_loop == False:
                old_adjacency_matrix_shape = adjacency_matrix.shape[0]

            ij = []

            adjacency_matrix = matcher.UpdateAdjacencyMatrix(
                adjacency_matrix,
                os.listdir(cfg.KF_DIR_BATCH / "cam0"),
                SEQUENTIAL_OVERLAP,
                first_colmap_loop,
            )

            #matcher.PlotAdjacencyMatrix(adjacency_matrix)
            keypoints, descriptors, images = import_local_features.ImportLocalFeature(
                cfg.DATABASE
            )
            true_indices = np.where(adjacency_matrix)

            d = new_n_keyframes - old_n_keyframes
            inverted_dict = {value: key for key, value in images.items()}

            # Matching cam0 at different epochs
            for l, m in zip(true_indices[0], true_indices[1]):
                if l > m and l > old_adjacency_matrix_shape - 1:
                    kfm1_name = keyframes_list.get_keyframe_by_id(m)._keyframe_name
                    kfm2_name = keyframes_list.get_keyframe_by_id(l)._keyframe_name
                    i = inverted_dict[f"cam0/{kfm2_name}"]
                    j = inverted_dict[f"cam0/{kfm1_name}"]
                    ij.append((i-1, j-1))

                    # Matching between different cameras at the same epoch
                    for c in range(1, cfg.N_CAMERAS):
                        i = inverted_dict[f"cam{c}/{kfm1_name}"]
                        ij.append((i-1, j-1))
            #for k in range(d):
            #    for c in range(len(cfg.N_CAMERAS)):
            #        i = old_n_keyframes + k
            #        j = i + c*d
            #        ij.append((i, j))


            for i, j in ij:
                im1 = images[j + 1]
                im2 = images[i + 1]
                matches_matrix = matcher.Matcher(
                    descriptors[j + 1].astype(float),
                    descriptors[i + 1].astype(float),
                    cfg.KORNIA_MATCHER,
                    cfg.RATIO_THRESHOLD,
                )
                if matches_matrix.shape[0] < cfg.LOCAL_FEAT_MIN_MATCHES:
                    continue
                db = db_colmap.COLMAPDatabase.connect(cfg.DATABASE)
                db.add_matches(int(j + 1), int(i + 1), matches_matrix)
                pts1 = keypoints[j + 1][matches_matrix[:, 0], :2].reshape((-1, 2))
                pts2 = keypoints[i + 1][matches_matrix[:, 1], :2].reshape((-1, 2))
                if pts1.shape[0] > 8:
                    if cfg.GEOMETRIC_VERIFICATION == "ransac":
                        F, mask = cv2.findFundamentalMat(
                            pts1,
                            pts2,
                            cv2.FM_RANSAC,
                            cfg.MAX_ERROR,
                            cfg.CONFIDENCE,
                            cfg.ITERATIONS,
                        )
                    elif cfg.GEOMETRIC_VERIFICATION == "pydegensac":
                        F, mask = pydegensac.findFundamentalMatrix(
                            pts1,
                            pts2,
                            px_th=cfg.MAX_ERROR,
                            conf=cfg.CONFIDENCE,
                            max_iters=cfg.ITERATIONS,
                            laf_consistensy_coef=-1,
                            error_type="sampson",
                            symmetric_error_check=True,
                            enable_degeneracy_check=True,
                        )
                    if mask.shape[0] > 8:
                        try:
                            if cfg.GEOMETRIC_VERIFICATION == "pydegensac":
                                mask = mask
                            elif cfg.GEOMETRIC_VERIFICATION == "ransac":
                                mask = mask[:, 0]
                            verified_matches_matrix = matches_matrix[mask, :]
                            db.add_two_view_geometry(
                                int(j + 1), int(i + 1), verified_matches_matrix
                            )
                        except:
                            logger.info("No valid geometry")
                    elif mask.shape[0] <= 8:
                        logger.info("N points < 8")
                db.commit()
                db.close()

        # if cfg.LOOP_CLOSURE_DETECTION == False:
        #    colmap.SequentialMatcher(database_path=cfg.DATABASE, loop_closure='0', overlap=str(SEQUENTIAL_OVERLAP), vocab_tree='')

        elif (
            cfg.LOOP_CLOSURE_DETECTION == True
            and cfg.LOCAL_FEAT_LOCAL_FEATURE == "RootSIFT"
        ):
            print("Joining this feature, please contact the author..\nQuit")
            quit()
            colmap.SequentialMatcher(
                database_path=cfg.DATABASE,
                loop_closure="1",
                overlap=str(SEQUENTIAL_OVERLAP),
                vocab_tree=cfg.VOCAB_TREE,
            )

        else:
            logger.info(
                "Not compatible option for loop closure detection. Currently only RootSIFT is supported for loop-closure detection. Quit."
            )
            quit()

        timer.update("SEQUENTIAL MATCHER")

        logger.info("MAPPER")

        colmap.Mapper(
            database_path=cfg.DATABASE,
            path_to_images=cfg.KF_DIR_BATCH,
            input_path=cfg.OUT_DIR_BATCH,
            output_path=cfg.OUT_DIR_BATCH,
            first_loop=first_colmap_loop,
        )
        
        if not os.path.exists(f"{cfg.OUT_DIR_BATCH}/0"):
            print("Failed Mapper. Reinitializing..")
            cfg = init.new_batch_solution()
            first_colmap_loop = True
            for im in processed_imgs:
                os.remove(cfg.CURRENT_DIR / im)

            # Reinit local feature extractor
            local_feat_conf = LocalFeatConfFile(cfg)
            local_feat_extractor = LocalFeatureExtractor(
                cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
            )

            # Setup logging level
            LOG_LEVEL = logging.INFO
            utils.Inizialization.setup_logger(LOG_LEVEL)
            logger = logging.getLogger("ColmapSLAM")

            # Initialize variables
            keyframes_list = KeyFrameList()
            processed_imgs = []
            pointer = 0
            delta = 0
            first_colmap_loop = True
            one_time = False
            reference_imgs = []

            # Setup keyframe selector
            kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
            keyframe_selector = KeyFrameSelector(
                keyframes_list=keyframes_list,
                last_keyframe_pointer=pointer,
                last_keyframe_delta=delta,
                keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
                kfs_method=cfg.KFS_METHOD,
                geometric_verification="pydegensac",
                local_feature=cfg.KFS_LOCAL_FEATURE,
                local_feature_cfg=kf_selection_detecor_config,
                n_features=cfg.KFS_N_FEATURES,
                realtime_viz=True,
                viz_res_path=SNAPSHOT_DIR,
                innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
                min_matches=cfg.MIN_MATCHES,
                error_threshold=cfg.RANSAC_THRESHOLD,
                iterations=cfg.RANSAC_ITERATIONS,
                n_camera=cfg.N_CAMERAS,

            )

            continue


        timer.update("MAPPER")

        logger.info("Convert model from binary to txt")
        subprocess.run(
            [
                str(cfg.COLMAP_EXE_PATH),
                "model_converter",
                "--input_path",
                cfg.OUT_DIR_BATCH / "0",
                "--output_path",
                cfg.OUT_DIR_BATCH,
                "--output_type",
                "TXT",
            ],
            stdout=subprocess.DEVNULL,
        )

        timer.update("MODEL CONVERSION")
        timer.print("COLMAP")

        # Export cameras
        lines, oriented_dict = export_cameras.ExportCameras(
            cfg.OUT_DIR_BATCH / "images.txt", keyframes_list
        )
        if cfg.DEBUG:
            with open(cfg.OUT_DIR_BATCH / "loc.txt", "w") as file:
                for line in lines:
                    file.write(line)
        print(cfg.DEBUG)

        # Keep track of sucessfully oriented frames in the current kfm_batch
        oriented_kfs_len = 0
        for keyframe in keyframes_list: # for image in kfm_batch
             if "cam0/" + keyframe.keyframe_name in list(oriented_dict.keys()):
                keyframe.set_oriented
                oriented_kfs_len += 1
                #print(keyframe.image_name, keyframe.image_id, keyframe.keyframe_id, keyframe.keyframe_name)
                #quit()
     
        ## Define new reference img (pointer)
        #print(list(oriented_dict.keys()))
        ##print(list(oriented_dict.keys()).glob(f"cam0/*.{cfg.IMG_FORMAT}"))
        #lista = list(oriented_dict.keys())
        ##matching_files = [file for file in lista if glob.fnmatch(file, "cam0/*")]
        ##matching_files = [file for file in lista if any(glob.glob("cam0/*.txt") == file)]
        #cam0_files = [item for item in lista if glob.fnmatch.fnmatch(item, 'cam0/*')]
        #print(cam0_files)
        #quit()
        #last_oriented_keyframe = np.max(list(oriented_dict.keys()))
        #keyframe_obj = keyframes_list.get_keyframe_by_id(last_oriented_keyframe)
        #n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
        #last_keyframe = keyframes_list.get_keyframe_by_id(n_keyframes - 1)
        #last_keyframe_img_id = last_keyframe.image_id
        #pointer = keyframe_obj.image_id  # pointer to the last oriented image
        #delta = last_keyframe_img_id - pointer
        delta = 0

        # Update dynamic window for sequential matching
        if delta != 0:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP + 2 * (
                n_keyframes - last_oriented_keyframe
            )
            if SEQUENTIAL_OVERLAP > MAX_SEQUENTIAL_OVERLAP:
                SEQUENTIAL_OVERLAP = MAX_SEQUENTIAL_OVERLAP
        else:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP

        oriented_dict_cam0 = {}
        for key in oriented_dict:
            cam, name = key.split("/", 1)
            id, extension = name.split(".", 1)
            id = int(id)
            if cam == "cam0":
                oriented_dict_cam0[id] = oriented_dict[key]
        oriented_dict = oriented_dict_cam0

        oriented_dict_list = list(oriented_dict.keys())
        oriented_dict_list.sort()
        total_kfs_number = len(kfrms)
        
        #oriented_kfs_len = len(oriented_dict_list)
        ori_ratio = oriented_kfs_len / total_kfs_number

        logger.info(
            f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}"
        )

        # Report SLAM solution in the reference system of the first image
        ref_img_id = oriented_dict_list[0]
        keyframe_obj = keyframes_list.get_keyframe_by_id(ref_img_id)
        keyframe_obj.slamX = 0.0
        keyframe_obj.slamY = 0.0
        keyframe_obj.slamZ = 0.0
        q0, t0 = oriented_dict[ref_img_id][1]
        t0 = t0.reshape((3, 1))
        q0_quat = Quaternion(q0)

        for keyframe_id in oriented_dict_list:
            keyframe_obj = keyframes_list.get_keyframe_by_id(keyframe_id)
            if keyframe_id == ref_img_id:
                pass
            else:
                qi, ti = oriented_dict[keyframe_id][1]
                ti = ti.reshape((3, 1))
                qi_quat = Quaternion(qi)
                ti_in_q0_ref = (
                    -np.dot((q0_quat * qi_quat.inverse).rotation_matrix, ti) + t0
                )
                keyframe_obj.slamX = ti_in_q0_ref[0, 0]
                keyframe_obj.slamY = ti_in_q0_ref[1, 0]
                keyframe_obj.slamZ = ti_in_q0_ref[2, 0]
                #print(keyframe_obj.slamX)
                #print(keyframe_obj.slamY)
                #print(keyframe_obj.slamZ)

        with open("./keyframes.pkl", "wb") as f:
            pickle.dump(keyframes_list, f)

        # Report 3D points in ref system of the first image
        with open(f"{cfg.OUT_DIR_BATCH}/points3D.txt", "r") as file:
            lines = file.readlines()

        data = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            values = line.split()
            x = float(values[1])
            y = float(values[2])
            z = float(values[3])
            p = np.array([[x], [y], [z]])
            p_trans = np.dot(q0_quat.rotation_matrix, p) + t0
            data.append([p_trans[0, 0], p_trans[1, 0], p_trans[2, 0]])

        with open("./points3D.pkl", "wb") as f:
            pickle.dump(np.array(data), f)

        kfm_batch = []
        first_colmap_loop = False

        # REINITIALIZE SLAM
        if (
            ori_ratio < cfg.MIN_ORIENTED_RATIO
            or total_kfs_number - oriented_kfs_len > cfg.NOT_ORIENTED_KFMS
        ):
            logger.info(
                f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}"
            )
            logger.info("Not enough oriented images")

            cfg = init.new_batch_solution()
            first_colmap_loop = True

            for im in processed_imgs:
                os.remove(cfg.CURRENT_DIR / im)

            # Reinit local feature extractor
            local_feat_conf = LocalFeatConfFile(cfg)
            local_feat_extractor = LocalFeatureExtractor(
                cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
            )

            # Setup logging level
            LOG_LEVEL = logging.INFO
            utils.Inizialization.setup_logger(LOG_LEVEL)
            logger = logging.getLogger("ColmapSLAM")

            # Initialize variables
            print("\n\nREINITIALIZE ..")
            keyframes_list = KeyFrameList()
            processed_imgs = []
            pointer = 0
            delta = 0
            first_colmap_loop = True
            one_time = False
            reference_imgs = []

            # Setup keyframe selector
            kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
            keyframe_selector = KeyFrameSelector(
                keyframes_list=keyframes_list,
                last_keyframe_pointer=pointer,
                last_keyframe_delta=delta,
                keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
                kfs_method=cfg.KFS_METHOD,
                geometric_verification="pydegensac",
                local_feature=cfg.KFS_LOCAL_FEATURE,
                local_feature_cfg=kf_selection_detecor_config,
                n_features=cfg.KFS_N_FEATURES,
                realtime_viz=True,
                viz_res_path=SNAPSHOT_DIR,
                innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
                min_matches=cfg.MIN_MATCHES,
                error_threshold=cfg.RANSAC_THRESHOLD,
                iterations=cfg.RANSAC_ITERATIONS,
                n_camera=cfg.N_CAMERAS,
            )

            continue

    timer_global.update(f"{len(kfrms)}")
    time.sleep(cfg.SLEEP_TIME)

if len(imgs) == 0:
    logger.info('No frames are detected. Check paths in config.ini')
    quit()

average_loop_time = timer_global.get_average_time()
total_time = timer_global.get_total_time()

logging.info(f"Average loop time: {average_loop_time:.4f} s")
logging.info(f"Total time: {total_time:.4f} s")

logger.info("Done.")
