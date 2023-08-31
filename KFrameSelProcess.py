import os
import cv2
import sys
import time
import glob
import shutil
import logging
import cProfile
import subprocess
import pydegensac

from lib import utils
from pathlib import Path
from lib.keyframe_selection import KeyFrameSelConfFile, KeyFrameSelector
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatConfFile, LocalFeatureExtractor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

new_images = []
class NewFileHandler(FileSystemEventHandler):

    def __init__(self, imgs, keyframe_selector, pointer, delta, screen, logger):
        self.imgs = imgs
        self.pointer = pointer
        self.delta = delta
        print('len(self.imgs)', len(self.imgs))
        self.keyframe_selector = keyframe_selector
        self.screen = screen
        self.logger = logger

    @staticmethod
    def display_image(image_path):
        image = cv2.imread(image_path)
        cv2.imshow("New Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_created(self, event):
        if event.is_directory:
            return
        print("New file created:", event.src_path)

        self.imgs.append(Path(event.src_path))
        self.imgs.sort()
        #imgs = sorted((cfg.IMGS_FROM_SERVER / "cam0").glob(f"*.{cfg.IMG_FORMAT}"))
        self.logger.info(f'len {len(self.imgs)}')
        newer_imgs = False  # To control that new keyframes are added
        processed = 0  # Number of processed images
        if len(self.imgs) < 1:
            pass
        if len(self.imgs) < 2:
            # Set first frame as keyframe
            img0 = self.imgs[self.pointer]
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
                        self.pointer + delta + 1,
                    )
                )
                return
        elif len(self.imgs) >= 2:
            if len(self.keyframe_selector.keyframes_list) == 0:
                # Set first frame as keyframe
                img0 = self.imgs[self.pointer]
                existing_keyframe_number = 0
                for c in range(cfg.N_CAMERAS):
                    shutil.copy(
                        img0.parent.parent / f"cam{c}" / img0.name,
                        cfg.KF_DIR_BATCH / f"cam{c}" / f"{utils.Id2name(existing_keyframe_number)}",
                    )
                camera_id = 1
                #if img0 not in self.keyframe_selector.keyframes_list.keyframes_names:
                self.keyframe_selector.keyframes_list.add_keyframe(
                    KeyFrame(
                        img0,
                        existing_keyframe_number,
                        utils.Id2name(existing_keyframe_number),
                        camera_id,
                        self.pointer + delta + 1,
                    )
                )
                print('start loop')
                ##################################################### HERE THE PROBLEM!!!!
            
            # Accumula ritardo quando si sta fermi
            for c, img in enumerate(self.imgs[self.pointer:]):
                # Decide if new images are valid to be added to the sequential matching
                # Only new images found in the target folder are processed.
                # No more than MAX_IMG_BATCH_SIZE imgs are processed.
                print('here')
                if img not in processed_imgs and c == 0:
                    processed_imgs.append(img)
                    processed += 1
                    continue
                if img in processed_imgs or processed >= cfg.MAX_IMG_BATCH_SIZE:
                    continue
                

                img1 = self.imgs[self.pointer]
                img2 = img
                self.logger.info(f"\nProcessing image pair ({img1}, {img2})")
                self.logger.info(f"pointer {self.pointer} c {c}")
                old_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
                (
                    keyframes_list,
                    self.pointer,
                    self.delta,
                    kfs_time,
                    conc
                ) = self.keyframe_selector.run(img1, img2)

                ## Set if new keyframes are added
                #new_n_keyframes = len(os.listdir(cfg.KF_DIR_BATCH / "cam0"))
                #if new_n_keyframes - old_n_keyframes > 0:
                #    newer_imgs = True
                #    kfm_batch.append(img.name)
                #    keyframe_obj = keyframes_list.get_keyframe_by_image_name(img)
                #    #with open('keyframes.txt', 'a') as kfm_imgs:
                #    #    kfm_imgs.write(f"{keyframe_obj._image_name},{cfg.KF_DIR_BATCH}/cam0/{keyframe_obj._keyframe_name}\n")
                processed_imgs.append(img)
                processed += 1

                #cv2.setWindowTitle("Keyframe Selection", 'ciao')
                #cv2.imshow("Keyframe Selection", conc)
                #if cv2.waitKey(1) == ord("q"):
                #    sys.exit()

                self.screen = []
                self.screen.append(conc)
                new_images.append(conc)

                logging.info(f'TIME FRAMES KFR SEL = {c*0.2}')


def KFrameSelProcess(
        cfg,
        keyframes_list, 
        pointer,
        delta,
        SNAPSHOT_DIR,
        processed_imgs,
        logger,
        kfm_batch,
        ):

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

    ### MANAGING FILE CHANGES IN DIRECTORIES
    imgs = sorted((cfg.IMGS_FROM_SERVER / "cam0").glob(f"*.{cfg.IMG_FORMAT}"))
    screen = []
    new_images.append(cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED))
    screen.append(cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED))
    event_handler = NewFileHandler(imgs, keyframe_selector, pointer, delta, screen, logger)
    observer = Observer()
    observer.schedule(event_handler, path=cfg.IMGS_FROM_SERVER / "cam0", recursive=False)
    observer.start()

    try:

        while True:
            #time.sleep(1)
            cv2.setWindowTitle("Keyframe Selection", 'ciao')
            cv2.imshow("Keyframe Selection", new_images[-1])
            if cv2.waitKey(1) == ord("q"):
                sys.exit()
            #print('len(imgs)', len(imgs))
            #print(imgs)


    
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()


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

    keyframes_list = KeyFrameList()


    ### CAMERA STREAM OR RUN SIMULATOR
    if cfg.USE_SERVER == True:
        stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
    else:
        stream_proc = subprocess.Popen(["python", "./simulator.py"])


    ### RUNNING TIME STATISTICS
    cProfile.run('''KFrameSelProcess(
        cfg,
        keyframes_list, 
        pointer,
        delta,
        SNAPSHOT_DIR,
        processed_imgs,
        logger,
        kfm_batch,
        )''')