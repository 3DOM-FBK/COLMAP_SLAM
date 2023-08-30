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

import os
import cv2
import time
import glob
import pickle
import shutil
import logging
import pydegensac
import subprocess
import multiprocessing
import numpy as np

from pyquaternion import Quaternion
from lib.colmapAPI import ColmapAPI
from lib.keyframe_selection import KeyFrameSelConfFile, KeyFrameSelector
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatConfFile, LocalFeatureExtractor
from lib import cameras
from pathlib import Path
from multiprocessing.managers import BaseManager

from lib import (
    ExtractCustomFeatures,
    database,
    db_colmap,
    export_cameras,
    import_local_features,
    matcher,
    utils,
)



def KFrameSelProcess(
        keyframes_list, 
        pointer,
        delta):

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

    while True:

        imgs = sorted((cfg.IMGS_FROM_SERVER / "cam0").glob(f"*.{cfg.IMG_FORMAT}"))
        newer_imgs = False  # To control that new keyframes are added
        processed = 0  # Number of processed images

        if len(imgs) < 1:
            pass

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
                    #with open('keyframes.txt', 'a') as kfm_imgs:
                    #    kfm_imgs.write(f"{keyframe_obj._image_name},{cfg.KF_DIR_BATCH}/cam0/{keyframe_obj._keyframe_name}\n")

                processed_imgs.append(img)
                processed += 1


def print_list_length(keyframes_list):
    logger = logging.getLogger(__name__)
    for i in range(5):
        time.sleep(5)
        for i in range(10):
            logger.info('ciao')

        #ob = keyframes_list.get_keyframe_by_id(1)
        #for k in keyframes_list.keyframes:
        print("File list length:",  len(keyframes_list.keyframes()))
        

class CustomManager(BaseManager):
    # nothing
    pass


if __name__ == '__main__':

    ### SETUP LOGGER
    # Setup logging level. Options are [INFO, WARNING, ERROR, CRITICAL]
    LOG_LEVEL = logging.INFO 
    utils.Inizialization.setup_logger(LOG_LEVEL)
    logger = logging.getLogger()
    logger.info('Setup logger finished')


    ### INITIALIZATION
    CFG_FILE = "config.ini"
    init = utils.Inizialization(CFG_FILE)
    cfg = init.inizialize()

    SEQUENTIAL_OVERLAP = cfg.SEQUENTIAL_OVERLAP
    MAX_SEQUENTIAL_OVERLAP = 50
    if cfg.SNAPSHOT == True:
        SNAPSHOT_DIR = './frames'
    elif cfg.SNAPSHOT == False:
        SNAPSHOT_DIR = None

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
    keypoints, descriptors, laf = {}, {}, {}


    ### CAMERA STREAM OR RUN SIMULATOR
    if cfg.USE_SERVER == True:
        stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
    else:
        stream_proc = subprocess.Popen(["python", "./simulator.py"])
    #stream_proc = subprocess.Popen(["python", "./lib/webcam.py"])


    ### RUN IN PARALLEL KEYFRAME SELECTION AND MAPPING
    #multiprocessing.freeze_support()

    # Register classes for variables to be shared between processes
    CustomManager.register('KeyFrameList', KeyFrameList)

    with CustomManager() as manager:
        keyframes_list = manager.KeyFrameList()
        update_process = multiprocessing.Process(
                                                target=KFrameSelProcess,
                                                args=(
                                                    keyframes_list,
                                                    pointer,
                                                    delta,
                                                    ))
        print_process = multiprocessing.Process(target=print_list_length, args=(keyframes_list,))
        update_process.start()
        print_process.start()
        update_process.join()
        print_process.join()

        logger.info('END')