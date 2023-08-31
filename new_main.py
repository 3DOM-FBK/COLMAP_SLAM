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

import KFrameSelProcess, MappingProcess



        

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
        newer_imgs = multiprocessing.Manager().Value('b', False)
        lock = multiprocessing.Manager().Lock()
        update_process = multiprocessing.Process(
                                                target=KFrameSelProcess.KFrameSelProcess,
                                                args=(
                                                    cfg,
                                                    keyframes_list,
                                                    pointer,
                                                    delta,
                                                    SNAPSHOT_DIR,
                                                    processed_imgs,
                                                    logger,
                                                    kfm_batch,
                                                    newer_imgs,
                                                    lock,
                                                    ))
        print_process = multiprocessing.Process(target=MappingProcess.MappingProcess, args=(
                                                                                                keyframes_list,
                                                                                                logger,
                                                                                                cfg,
                                                                                                newer_imgs,
                                                                                                first_colmap_loop,
                                                                                                lock,
                                                                                                SEQUENTIAL_OVERLAP,
                                                                                                adjacency_matrix,
                                                                                                keypoints,
                                                                                                descriptors,
                                                                                                laf,
                                                                                                ))
        update_process.start()
        print_process.start()
        update_process.join()
        print_process.join()

        logger.info('END')