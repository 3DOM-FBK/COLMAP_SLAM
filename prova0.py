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
from lib.local_features import LocalFeatConfFile, LocalFeatureExtractor
from lib import cameras
from pathlib import Path
from multiprocessing.managers import BaseManager
from typing import List, Union
from lib import (
    ExtractCustomFeatures,
    database,
    db_colmap,
    export_cameras,
    import_local_features,
    matcher,
    utils,
)




class KeyFrameList:
    def __init__(self):
        self._keyframes = []
        self._current_idx = 0

    def __len__(self):
        return len(self._keyframes)

    def __getitem__(self, keyframe_id):
        return self._keyframes[keyframe_id]

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= len(self._keyframes):
            raise StopIteration
        cur = self._current_idx
        self._current_idx += 1
        return self._keyframes[cur]

    def keyframes(self):
        return self._keyframes

    def add_keyframe(self, keyframe) -> None:
        self._keyframes.append(keyframe)

    def get_keyframe_by_name(self, keyframe_name: str):
        for keyframe in self._keyframes:
            if keyframe._keyframe_name == keyframe_name:
                return keyframe
        return None

class KeyFrame:
    def __init__(self, image_name : Union[str, Path], keyframe_id, keyframe_name, camera_id, image_id, manager2):
        self._image_name = image_name
        self._image_id = image_id
        self._keyframe_id = keyframe_id
        self._keyframe_name = keyframe_name
        self._camera_id = camera_id
        self._oriented = False
        self.n_keypoints = 0
        self.GPSLatitude = "-"
        self.GPSLongitude = "-"
        self.GPSAltitude = "-"
        self.enuX = "-"
        self.enuY = manager2.Value('b', 0)
        self.enuZ = "-"
        self.slamX = "-"
        self.slamY = "-"
        self.slamZ = "-"
        self.slave_cameras_POS = {}
        self.time_last_modification = time.time()



def KFrameSelProcess(
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
        shared_list,
        manager2,
        ):

    for i in range(5):
        time.sleep(0.25)
        #with lock:
        #    keyframes_list.add_keyframe(keyframes_list.KeyFrame(i,i,i,i,i))
        #    c = keyframes_list.get_keyframe_by_name(i)
        #    print('KFrameSelProcess: added keyframe', 'c', i, id(c))
        #    
        #    #newer_imgs.value = False
        with lock:
            shared_list.append(
                KeyFrame(i,i,i,i,i, manager2)
            )
            for obj in shared_list:
                print('obj.slamY', obj.slamY)


def MappingProcess(
        keyframes_list, 
        logger, 
        cfg, 
        newer_imgs, 
        first_colmap_loop, 
        lock, SEQUENTIAL_OVERLAP, 
        adjacency_matrix, 
        keypoints, 
        descriptors, 
        laf, 
        init, 
        SNAPSHOT_DIR, 
        processed_imgs, 
        shared_list,
        manager2,
        ):
    
    for i in range(5):
        time.sleep(1)
        with lock:
            shared_list[0].slamY = 9
        ##with lock:
        #    #o = keyframes_list.get_keyframe_by_name(0)
        #    o = keyframes_list.keyframes()[0]
        #    print(len(keyframes_list.keyframes()))
        #    print('MappingProcess', 'o', i, id(o))
        #    #o.slamY = 7
        #    #newer_imgs.value = True
        #    
        ##with lock:
        #    time.sleep(5)
        #    o = keyframes_list.keyframes()[0]
        #    print(len(keyframes_list.keyframes()))
        #    #o = keyframes_list.get_keyframe_by_name(0)
        #    print('MappingProcess', 'o', i, id(o))
        #    #print('o', i, id(o))
        #    #print(f'Mapping: slamY {o.slamY}')
        #    #print(newer_imgs.value)

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


    kfm_batch = []
    pointer = 0  # pointer points to the last oriented image
    delta = 0  # delta is equal to the number of processed but not oriented imgs
    first_colmap_loop = True
    one_time = False  # It becomes true after the first batch of images is oriented
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
    #CustomManager.register('KeyFrame', KeyFrame)
    CustomManager.register('KeyFrameList', KeyFrameList)

    with CustomManager() as manager:
        manager2 = multiprocessing.Manager()
        shared_list = manager2.list()
        keyframes_list = manager.KeyFrameList()
        newer_imgs = manager2.Value('b', False)
        lock = manager2.Lock()
        processed_imgs =manager2.list()
        update_process = multiprocessing.Process(
                                                target=KFrameSelProcess,
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
                                                    shared_list,
                                                    manager2,
                                                    ))
        print_process = multiprocessing.Process(target=MappingProcess, args=(
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
                                                                                                init,
                                                                                                SNAPSHOT_DIR,
                                                                                                processed_imgs,
                                                                                                shared_list,
                                                                                                manager2,
                                                                                                ))
        
        update_process.start()
        #print_process.start()
        
        update_process.join()
        #print_process.join()

        logger.info('END')