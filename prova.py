#import os
#import time
#import multiprocessing
#
## Function to update the file list
#def update_file_list(folder_path, file_lists):
#    while True:
#        files = os.listdir(folder_path)
#        file_lists.append(files)
#        time.sleep(5)
#
## Function to print the length of the file list
#def print_list_length(file_lists):
#    while True:
#        print("File list length:", len(file_lists[-1]))
#        time.sleep(1)
#
#if __name__ == "__main__":
#    folder_path = "./imgs/cam0"
#    file_lists = multiprocessing.Manager().list()
#
#    update_process = multiprocessing.Process(target=update_file_list, args=(folder_path, file_lists))
#    print_process = multiprocessing.Process(target=print_list_length, args=(file_lists,))
#
#    update_process.start()
#    print_process.start()
#
#    update_process.join()
#    print_process.join()


import logging
import os
import pickle
import shutil
import subprocess
import time
import glob
import multiprocessing

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

from multiprocessing.managers import BaseManager

## Configuration file
#CFG_FILE = "config.ini"
#
### Setup logging level. Options are [INFO, WARNING, ERROR, CRITICAL]
##LOG_LEVEL = logging.INFO
### utils.Inizialization.setup_logger(LOG_LEVEL)
### logger = logging.getLogger("ColmapSLAM")
##if os.path.exists('debug.log'):
##    os.remove('debug.log')
##logging.basicConfig(filename='debug.log', level=logging.DEBUG,
##                    format='%(asctime)s - %(levelname)s - %(message)s')
##console_handler = logging.StreamHandler()
##console_handler.setLevel(logging.DEBUG)  # Set the desired level for console output
##console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
##console_handler.setFormatter(console_formatter)
##logging.getLogger().addHandler(console_handler)
##logger = logging.getLogger()
#
## Inizialize COLMAP SLAM problem
#init = utils.Inizialization(CFG_FILE)
#cfg = init.inizialize()
#cfg.PLOT_TRJECTORY = False # Please keep PLOT_TRAJECTORY = False.
## It is used to plot an additional trajectory (matplotlib) but for now shows rarely a bug (multiple plots).
## In any case the plot from opencv will show
#
## Initialize variables
#SEQUENTIAL_OVERLAP = cfg.SEQUENTIAL_OVERLAP
#MAX_SEQUENTIAL_OVERLAP = 50
#if cfg.SNAPSHOT == True:
#    SNAPSHOT_DIR = './frames'
#elif cfg.SNAPSHOT == False:
#    SNAPSHOT_DIR = None
#keyframes_list = KeyFrameList()
#processed_imgs = []
#kfm_batch = []
#pointer = 0  # pointer points to the last oriented image
#delta = 0  # delta is equal to the number of processed but not oriented imgs
#first_colmap_loop = True
#one_time = False  # It becomes true after the first batch of images is oriented
## The first batch of images define the reference system.
## At following epochs the photogrammetric model will be reported in this ref system.
#reference_imgs = []
#adjacency_matrix = None
#keypoints, descriptors, laf = {}, {}, {}
#
## Setup keyframe selector
#kf_selection_detecor_config = KeyFrameSelConfFile(cfg)
#keyframe_selector = KeyFrameSelector(
#    keyframes_list=keyframes_list,
#    last_keyframe_pointer=pointer,
#    last_keyframe_delta=delta,
#    keyframes_dir=cfg.KF_DIR_BATCH / "cam0",
#    kfs_method=cfg.KFS_METHOD,
#    geometric_verification="pydegensac",
#    local_feature=cfg.KFS_LOCAL_FEATURE,
#    local_feature_cfg=kf_selection_detecor_config,
#    n_features=cfg.KFS_N_FEATURES,
#    realtime_viz=True,
#    viz_res_path=SNAPSHOT_DIR,
#    innovation_threshold_pix=cfg.INNOVATION_THRESH_PIX,
#    min_matches=cfg.MIN_MATCHES,
#    error_threshold=cfg.RANSAC_THRESHOLD,
#    iterations=cfg.RANSAC_ITERATIONS,
#    n_camera=cfg.N_CAMERAS,
#)
#
## Setup local feature to use on keyframes
#local_feat_conf = LocalFeatConfFile(cfg)
#local_feat_extractor = LocalFeatureExtractor(
#    cfg.LOCAL_FEAT_LOCAL_FEATURE, local_feat_conf, cfg.LOCAL_FEAT_N_FEATURES, cfg.CAM, cfg.DATABASE,
#)
#
## If the camera coordinates are known from other sensors than gnss,
## they can be stores in camera_coord_other_sensors dictionary and used
## to scale the photogrammetric model
#camera_coord_other_sensors = {}
#if cfg.USE_EXTERNAL_CAM_COORD == True:
#    with open(cfg.CAMERA_COORDINATES_FILE, "r") as gt_file:
#        lines = gt_file.readlines()
#        for line in lines[2:]:
#            id, x, y, z, _ = line.split(" ", 4)
#            camera_coord_other_sensors[id] = (x, y, z)
#
## Stream of input data
#if cfg.USE_SERVER == True:
#    stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
#else:
#    stream_proc = subprocess.Popen(["python", "./simulator.py"])
#
#
##stream_proc = subprocess.Popen(["python", "./lib/webcam.py"])
#
## Set-up plotqq
## create_plot()
#if cfg.PLOT_TRJECTORY:
#    plot_proc = subprocess.Popen(["python", "./plot.py"])
#
## Initialize COLMAP API
#colmap = ColmapAPI(str(cfg.COLMAP_EXE_PATH))



def k_frame(frames_dir, keyframes_list):
    for ciao in range (2):
        
        imgs = os.listdir(frames_dir)
        keyframes_list.add_keyframe(
                    KeyFrame(
                        f'img{ciao}',
                        ciao,
                        ciao,
                        ciao,
                        ciao,
                    )
                )      
        time.sleep(3)  

        #elif len(imgs) < 2:
        #    # Set first frame as keyframe
        #    img0 = imgs[pointer]
        #    existing_keyframe_number = 0
        #    for c in range(cfg.N_CAMERAS):
        #        shutil.copy(
        #            img0.parent.parent / f"cam{c}" / img0.name,
        #            cfg.KF_DIR_BATCH / f"cam{c}" / f"{utils.Id2name(existing_keyframe_number)}",
        #        )
        #    camera_id = 1
        #    if img0 not in keyframes_list.keyframes_names:
        #        keyframes_list.add_keyframe(
        #            KeyFrame(
        #                img0,
        #                existing_keyframe_number,
        #                utils.Id2name(existing_keyframe_number),
        #                camera_id,
        #                pointer + delta + 1,
        #            )
        #        )
#
        #else:
        #    time.sleep(1)
        #    print('ok')
        #    
#
        #    #for c, img in enumerate(imgs):
        #    #    if img not in processed_imgs and c == 0:
        #    #        processed_imgs.append(img)
        #    #        processed += 1
        #    #        continue
        #    #    if img in processed_imgs or processed >= cfg.MAX_IMG_BATCH_SIZE:
        #    #        continu
        #    #    img1 = imgs[pointer]
        #    #    img2 = im
        #    #    logger.info(f"\nProcessing image pair ({img1}, {img2})")
        #    #    logger.info(f"pointer {pointer} c {c}"
        #    #    old_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0")
        #    #    (
        #    #        keyframes_list,
        #    #        pointer,
        #    #        delta,
        #    #        kfs_time,
        #    #    ) = keyframe_selector.run(img1, img2
        #    #    # Set if new keyframes are added
        #    #    new_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
        #    #    if new_n_keyframes - old_n_keyframes > 0:
        #    #        newer_imgs = True
        #    #        kfm_batch.append(img.name)
        #    #        keyframe_obj = keyframes_list.get_keyframe_by_image_name(img)
        #    #        with open('keyframes.txt', 'a') as kfm_imgs:
        #    #            kfm_imgs.write(f"{keyframe_obj._image_name},{cfg.KF_DIR_BATCH}/cam0/{keyframe_obj._keyframe_name}\n")

def print_list_length(keyframes_list):
    for i in range(5):
        time.sleep(4)

        ob = keyframes_list.get_keyframe_by_image_name('img1')
        #for k in keyframes_list.keyframes:
        print("File list length:",  ob.image_name)
        

class CustomManager(BaseManager):
    # nothing
    pass



if __name__ == '__main__':
    
    #multiprocessing.freeze_support()
    CustomManager.register('KeyFrameList', KeyFrameList)
    with CustomManager() as manager:
        # create a shared custom class instance
        keyframes_list = manager.KeyFrameList()
        #print(keyframes_list)


        update_process = multiprocessing.Process(target=k_frame, args=("imgs/cam0", keyframes_list,))
        print_process = multiprocessing.Process(target=print_list_length, args=(keyframes_list,))

        update_process.start()
        print_process.start()

        update_process.join()
        print_process.join()
        print('ok')