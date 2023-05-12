import configparser
import logging
import os
import pickle
import shutil
import subprocess
import time
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import plotly.graph_objs as go
import numpy as np
import piexif

from pathlib import Path
from easydict import EasyDict as edict
from matplotlib import interactive
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
from scipy import linalg
from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots
from lib.keyframe_selection import KeyFrameSelector, KeyFrameSelConfFile
from lib.keyframes import KeyFrame, KeyFrameList
from lib.colmapAPI import ColmapAPI
from lib.local_features import LocalFeatureExtractor, LocalFeatConfFile
from lib import (
    ExtractCustomFeatures,
    database,
    export_cameras,
    utils,
)


def colmap_process() -> bool:
    pass


def read_camera_eo(fname: str = "./keyframes.pkl") -> tuple:
    if not os.path.exists(fname):
        return

    X = []
    Y = []
    Z = []
    with open(fname, "rb") as f:
        my_list = pickle.load(f)
        for obj in my_list:
            if obj.slamX != "-":
                X.append(float(obj.slamX))
            if obj.slamY != "-":
                Y.append(float(obj.slamY))
            if obj.slamZ != "-":
                Z.append(float(obj.slamZ))

    return X, Y, Z


def create_plot():
    global camera_pos_trace, camera_data_trace, fig
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title_text="Camera Trajectory",
    )

    # Add the camera position trace to the figure
    camera_pos_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="red")
    )
    fig.add_trace(camera_pos_trace)

    # Add the camera data trace to the figure
    camera_data_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="lines", line=dict(color="blue", width=2)
    )
    fig.add_trace(camera_data_trace)

    # Set the plot limits and viewpoint
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        ),
        scene_aspectmode="cube",
    )


def update_plot():
    # Read the camera data
    X, Y, Z = read_camera_eo("./keyframes.pkl")

    # Update the camera position trace
    camera_pos_trace.x = [X[-1]]
    camera_pos_trace.y = [Y[-1]]
    camera_pos_trace.z = [Z[-1]]

    # Update the camera data trace
    camera_data_trace.x = X
    camera_data_trace.y = Y
    camera_data_trace.z = Z

    # Set the plot title
    fig.update_layout(title_text=f"Camera Trajectory (n={len(X)})")

# Configuration file
CFG_FILE = "config.ini"

# Setup logging level
LOG_LEVEL = logging.INFO
utils.Inizialization.setup_logger(LOG_LEVEL)
logger = logging.getLogger("ColmapSLAM")

# Inizialize COLMAP SLAM problem
init = utils.Inizialization(CFG_FILE)
cfg = init.inizialize()

# Initialize variables
keyframes_list = KeyFrameList()
processed_imgs = []
oriented_imgs_batch = []
pointer = 0  # pointer points to the last oriented image
delta = 0  # delta is equal to the number of processed but not oriented imgs
first_colmap_loop = True
one_time = False  # It becomes true after the first batch of images is oriented
# The first batch of images define the reference system.
# At following epochs the photogrammetric model will be reported in this ref system.
reference_imgs = []
SEQUENTIAL_OVERLAP = cfg.SEQUENTIAL_OVERLAP
MAX_SEQUENTIAL_OVERLAP = 15

# Setup keyframe selector
kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
keyframe_selector = KeyFrameSelector(
    keyframes_list=keyframes_list,
    last_keyframe_pointer=pointer,
    last_keyframe_delta=delta,
    keyframes_dir=cfg.KF_DIR_BATCH,
    kfs_method=cfg.KFS_METHOD,
    geometric_verification="pydegensac",
    local_feature=cfg.KFS_LOCAL_FEATURE,
    local_feature_cfg=kf_selection_detecor_config,
    n_features=cfg.KFS_N_FEATURES,
    realtime_viz=True,
    viz_res_path=None,
)

# Setup local feature to use on keyframes
local_feat_conf = LocalFeatConfFile(cfg)
local_feat_extractor = LocalFeatureExtractor(
    cfg.LOCAL_FEAT_LOCAL_FEATURE, 
    local_feat_conf,
    cfg.LOCAL_FEAT_N_FEATURES,
    cfg.CAM0
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
    stream_proc = subprocess.Popen(["python3", "./simulator.py"])

# Set-up plotqq
# create_plot()
if cfg.PLOT_TRJECTORY:
    plot_proc = subprocess.Popen(["python3", "./plot.py"])

# Initialize COLMAP API
colmap = ColmapAPI(str(cfg.COLMAP_EXE_PATH))



# MAIN LOOP
timer_global = utils.AverageTimer(logger=logger)
while True:
    # Get sorted image list available in imgs folders
    imgs = sorted(cfg.IMGS_FROM_SERVER.glob(f"*.{cfg.IMG_FORMAT}"))

    # If using the simulator, check if process is still alive, otherwise quit
    if cfg.USE_SERVER == False:
        if stream_proc.poll() is not None:
            logging.info("Simulator completed.")
            if cfg.PLOT_TRJECTORY:
                plot_proc.kill()
            break
    else:
        # Make exit condition when using server
        pass

    img_batch = []

    newer_imgs = False  # To control that new keyframes are added
    processed = 0  # Number of processed images

    # Keyframe selection
    if len(imgs) < 1:
        continue

    if len(imgs) < 2:
        # Set first frame as keyframe
        img0 = imgs[pointer]
        existing_keyframe_number = 0
        shutil.copy(
            img0,
            cfg.KF_DIR_BATCH / f"{utils.Id2name(existing_keyframe_number)}",
        )
        camera_id = 1
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

            old_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH))

            (
                keyframes_list,
                pointer,
                delta,
                kfs_time,
            ) = keyframe_selector.run(img1, img2)

            # Set if new keyframes are added
            new_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH))
            if new_n_keyframes - old_n_keyframes > 0:
                newer_imgs = True
                img_batch.append(img)
                keyframe_obj = keyframes_list.get_keyframe_by_image_name(img)

                ## Load exif data and store GNSS position if present
                ## or load camera cooridnates from other sensors
                ## Exif gnss data works properly only when directions are N and E
                #exif_data = []
                #try:
                #    exif_data = piexif.load(str(img2))
                #except:
                #    logger.error(
                #        "Error loading exif data. Image file could be corrupted."
                #    )
#
                #if exif_data != [] and len(exif_data["GPS"].keys()) != 0:
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
                #elif exif_data != [] and img2 in camera_coord_other_sensors.keys():
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
    kfrms = os.listdir(cfg.KF_DIR_BATCH)
    kfrms.sort()

    if len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs == True:
        timer = utils.AverageTimer(logger=logger)
        logger.info(f"DYNAMIC MATCHING WINDOW: {cfg.SEQUENTIAL_OVERLAP}")

        if first_colmap_loop == True:
            logger.info('Initialize an empty database')
            colmap.CreateEmptyDatabase(cfg.DATABASE)
            timer.update("DATABASE INITIALIZATION")

        logger.info('Feature extraction')
        if cfg.LOCAL_FEAT_LOCAL_FEATURE != 'RootSIFT':
            local_feat_extractor.run(cfg.DATABASE, cfg.KF_DIR_BATCH, cfg.IMG_FORMAT)
        elif cfg.LOCAL_FEAT_LOCAL_FEATURE == 'RootSIFT':
            colmap.ExtractRootSiftFeatures(database_path=cfg.DATABASE, path_to_images=cfg.KF_DIR_BATCH, first_loop=first_colmap_loop, max_n_features=cfg.LOCAL_FEAT_N_FEATURES)
        timer.update("FEATURE EXTRACTION")

        logger.info('Sequential matcher')
        if cfg.LOOP_CLOSURE_DETECTION == False:
            colmap.SequentialMatcher(database_path=cfg.DATABASE, loop_closure='0', overlap=str(SEQUENTIAL_OVERLAP), vocab_tree='')
        elif cfg.LOOP_CLOSURE_DETECTION == True and cfg.CUSTOM_FEATURES == False:
            colmap.SequentialMatcher(database_path=cfg.DATABASE, loop_closure='1', overlap=str(SEQUENTIAL_OVERLAP), vocab_tree=cfg.VOCAB_TREE)
        else:
            print("Not compatible option for loop closure detection. Quit.")
            quit()
        timer.update("SEQUENTIAL MATCHER")

        print('MAPPER')
        colmap.Mapper(database_path=cfg.DATABASE, path_to_images=cfg.KF_DIR_BATCH, input_path=cfg.OUT_DIR_BATCH, output_path=cfg.OUT_DIR_BATCH, first_loop=first_colmap_loop)
        timer.update("MAPPER")

        logger.info('Convert model from binary to txt')
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

        # Keep track of sucessfully oriented frames in the current img_batch
        for image in img_batch:
            keyframe_obj = keyframes_list.get_keyframe_by_image_name(image)
            if keyframe_obj.keyframe_id in list(oriented_dict.keys()):
                oriented_imgs_batch.append(image)
                keyframe_obj.set_oriented()

        # Define new reference img (pointer)
        last_oriented_keyframe = np.max(list(oriented_dict.keys()))
        keyframe_obj = keyframes_list.get_keyframe_by_id(last_oriented_keyframe)
        n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH))
        last_keyframe = keyframes_list.get_keyframe_by_id(n_keyframes - 1)
        last_keyframe_img_id = last_keyframe.image_id
        pointer = keyframe_obj.image_id  # pointer to the last oriented image
        delta = last_keyframe_img_id - pointer

        # Update dynamic window for sequential matching
        if delta != 0:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP + 2 * (
                n_keyframes - last_oriented_keyframe
            )
            if SEQUENTIAL_OVERLAP > MAX_SEQUENTIAL_OVERLAP:
                SEQUENTIAL_OVERLAP = MAX_SEQUENTIAL_OVERLAP
        else:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP

        oriented_dict_list = list(oriented_dict.keys())
        oriented_dict_list.sort()

        total_kfs_number = len(kfrms)
        oriented_kfs_len = len(oriented_dict_list)
        ori_ratio = oriented_kfs_len / total_kfs_number


        print(f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}")

        # Report SLAM solution in the reference system of the first image
        #if first_colmap_loop == True:
        ref_img_id = oriented_dict_list[0]
        keyframe_obj = keyframes_list.get_keyframe_by_id(ref_img_id)
        keyframe_obj.slamX = 0.0
        keyframe_obj.slamY = 0.0
        keyframe_obj.slamZ = 0.0
        q0, t0 = oriented_dict[ref_img_id][2]
        t0 = t0.reshape((3,1))
        q0_quat = Quaternion(q0)

        #else:
        for keyframe_id in oriented_dict_list:
            keyframe_obj = keyframes_list.get_keyframe_by_id(keyframe_id)
            if keyframe_id == ref_img_id:
                pass
            else:
                qi, ti = oriented_dict[keyframe_id][2]
                ti = ti.reshape((3,1))
                qi_quat = Quaternion(qi)
                ti_in_q0_ref = -np.dot((q0_quat * qi_quat.inverse).rotation_matrix, ti) + t0
                keyframe_obj.slamX = ti_in_q0_ref[0, 0]
                keyframe_obj.slamY = ti_in_q0_ref[1, 0]
                keyframe_obj.slamZ = ti_in_q0_ref[2, 0]

        with open("./keyframes.pkl", "wb") as f:
            pickle.dump(keyframes_list, f)

        img_batch = []
        oriented_imgs_batch = []
        first_colmap_loop = False

        # REINITIALIZE SLAM
        if ori_ratio < cfg.MIN_ORIENTED_RATIO or total_kfs_number - oriented_kfs_len > cfg.NOT_ORIENTED_KFMS:
            print(f"Total keyframes: {total_kfs_number}; Oriented keyframes: {oriented_kfs_len}; Ratio: {ori_ratio}")
            print('Not enough oriented images')

            cfg = init.new_batch_solution()
            first_colmap_loop = True
            for im in processed_imgs:
                os.remove(cfg.CURRENT_DIR / im)

            # Setup logging level
            LOG_LEVEL = logging.INFO
            utils.Inizialization.setup_logger(LOG_LEVEL)
            logger = logging.getLogger("ColmapSLAM")

            # Initialize variables
            keyframes_list = KeyFrameList()
            processed_imgs = []
            oriented_imgs_batch = []
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
                keyframes_dir=cfg.KF_DIR_BATCH,
                kfs_method=cfg.KFS_METHOD,
                geometric_verification="pydegensac",
                local_feature=cfg.KFS_LOCAL_FEATURE,
                local_feature_cfg=kf_selection_detecor_config,
                n_features=cfg.KFS_N_FEATURES,
                realtime_viz=True,
                viz_res_path=None,
            )
            
            continue
 
    timer_global.update(f"{len(kfrms)}")
    time.sleep(cfg.SLEEP_TIME)

average_loop_time = timer_global.get_average_time()
total_time = timer_global.get_total_time()
timer_global.print("Timer global")

logging.info(f"Average loop time: {average_loop_time:.4f} s")
logging.info(f"Total time: {total_time:.4f} s")

print("Done.")