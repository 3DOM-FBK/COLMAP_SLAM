import os
import time
import logging

from config import config
from src.frames_reader import FramesReader
from src.keyframe_selection import KeyframeSelector
from pathlib import Path
from src.utils import timer
from multiprocessing import Pool

CFG_FILE = "config.ini"
LOG_LEVEL = logging.DEBUG # Setup logging level. Options are [INFO, WARNING, ERROR, CRITICAL, DEBUG]


def SetLogger() -> logging.Logger:
    if os.path.exists('debug.log'):
        os.remove('debug.log')
    logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
    logger = logging.getLogger()
    return logger

def ReadFrames(config: dict) -> None:
    #imgs = SortByTimeStamp(imgs)
    reader = FramesReader(
        start = config["frames_reader"]["start"],  
        end = config["frames_reader"]["end"], 
        show_camera_flow = config["frames_reader"]["show_camera_flow"]
        )
    reader.simulator(
        frames_dir = Path(config["frames_reader"]["path_to_frames_dir"]),
        output_dir = Path("./frames"),
        ext = ".jpg",
        step = config["frames_reader"]["step"],
        sleep = config["frames_reader"]["sleep"],
        equalize = config["frames_reader"]["equalize"],
        n_camera = config["calibration"]["n_camera"],
    )

def KeyframeSelection(config: dict) -> None:
    keyframe_selector = KeyframeSelector(
        frames_dir = Path('./frames'),
        keyframes_dir = Path('./keyframes'),
        show = config["keyframe_selector"]["show"],  
        n_camera = config["calibration"]["n_camera"],
        max_keypoints = config["keyframe_selector"]["max_keypoints"]     
    )
    keyframe_selector.run()

if __name__ == "__main__":

    logger = SetLogger()
    logger.info(f"COLMAP SLAM")
    timer = timer.AverageTimer(logger=logger)
    KeyframeSelection(config);quit()
    with Pool(processes=2) as pool:
        result1 = pool.apply_async(ReadFrames, args=(config,))
        result2 = pool.apply_async(KeyframeSelection)

        # Wait for all tasks to complete
        result1.get()
        result2.get()

    #init = utils.Inizialization(CFG_FILE)
    #cfg = init.inizialize()
    #timer.update("set configuration")
    #timer.print("READ CONFIGURATION")



