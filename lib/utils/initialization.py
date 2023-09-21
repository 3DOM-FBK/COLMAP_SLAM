import configparser
import logging
import os
import shutil
import json
from datetime import date, datetime
from pathlib import Path

from easydict import EasyDict as edict


# Import conf options
# TODO: centralized all options in config.ini file and cfg dictionary
# TODO: change parameter names to lowercase (both in Inizialization class and main)
# TODO: organize parameters in tree structure of dictionaries (or similar) to easily pass to function set of parameters
# TODO: add checks on parameters
class Inizialization:
    """Parse the configuration file and initialize the object's properties"""

    def __init__(self, cfg_file: str) -> None:
        """
        Args:
            cfg_file (str): Path to the configuration file
        """
        self.cfg_file = cfg_file

    #@staticmethod
    #def setup_logger(log_level) -> None:
    #    log_dir = Path("logs")
    #    log_dir.mkdir(exist_ok=True)
    #    today = date.today()
    #    now = datetime.now()
    #    current_date = f"{today.strftime('%Y_%m_%d')}_{now.strftime('%H:%M')}"
    #    log_file = log_dir / f"colmapslam_{current_date}.log"
    #    logging.basicConfig(
    #        level=log_level,
    #        format="%(asctime)s | %(levelname)s | %(message)s",
    #        datefmt="%Y-%m-%d %H:%M:%S",
    #        handlers=[
    #            logging.FileHandler(log_file),
    #            logging.StreamHandler(),
    #        ],
    #    )

    def parse_config_file(self) -> edict:
        """
        Parse the configuration file and store the values in a dictionary

        Returns:
            dict: A dictionary containing all the configuration options
        """
        config = configparser.ConfigParser()
        config.read(self.cfg_file, encoding="utf-8")

        cfg = edict({})

        # DEFAULT
        cfg.PLOT_TRJECTORY = config["DEFAULT"].getboolean("PLOT_TRJECTORY")
        cfg.SNAPSHOT = config["DEFAULT"].getboolean("SNAPSHOT")
        cfg.OS = config["DEFAULT"]["OS"]
        cfg.FORCE_PROCESS_ALL_FRAMES = config["DEFAULT"].getboolean("FORCE_PROCESS_ALL_FRAMES")
        cfg.RESIZE_FACTOR = int(config["DEFAULT"]["RESIZE_FACTOR"])
        cfg.USE_SERVER = config["DEFAULT"].getboolean("USE_SERVER")
        cfg.LAUNCH_SERVER_PATH = Path(config["DEFAULT"]["LAUNCH_SERVER_PATH"])
        cfg.DEBUG = config["DEFAULT"].getboolean("DEBUG")
        cfg.MAX_IMG_BATCH_SIZE = int(config["DEFAULT"]["MAX_IMG_BATCH_SIZE"])
        cfg.SIMULATOR_SLEEP_TIME = float(config["DEFAULT"]["SIMULATOR_SLEEP_TIME"])
        cfg.SLEEP_TIME = float(config["DEFAULT"]["SLEEP_TIME"])
        cfg.LOOP_CYCLES = int(config["DEFAULT"]["LOOP_CYCLES"])
        cfg.COLMAP_EXE_DIR = Path(config["DEFAULT"]["COLMAP_EXE_DIR"])
        cfg.IMGS_FROM_SERVER = Path(
            config["DEFAULT"]["IMGS_FROM_SERVER"]
        )  # Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs")
        cfg.STEP = config["DEFAULT"]["STEP"]
        cfg.IMG_FORMAT = config["DEFAULT"]["IMG_FORMAT"]
        cfg.INITIAL_SEQUENTIAL_OVERLAP = int(
            config["DEFAULT"]["INITIAL_SEQUENTIAL_OVERLAP"]
        )
        cfg.SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP
        cfg.MIN_ORIENTED_RATIO = float(config["DEFAULT"]["MIN_ORIENTED_RATIO"])
        cfg.NOT_ORIENTED_KFMS = int(config["DEFAULT"]["NOT_ORIENTED_KFMS"])

        # CALIBRATION DATA
        cfg.N_CAMERAS = int(config["CALIBRATION"]["N_CAMERAS"])
        cfg.CAM = []
        for c in range(cfg.N_CAMERAS):
            cam_type, width, height, dist_par = config["CALIBRATION"][f"CAM{c}"].strip().split(",", 3)
            data_camera = {
                "cam_type" : cam_type,
                "width" : width,
                "height" : height,
                "dist_par": dist_par,
            }
            cfg.CAM.append(data_camera)
        
        if cfg.N_CAMERAS > 1:
            cfg.BASELINE_CAM0_CAM1 = float(config["CALIBRATION"]["BASELINE_CAM0_CAM1"])
        
        with open("./lib/camera_models.json", 'r') as json_file:
            json_data = json_file.read()
            json_data = json.loads(json_data)
            for key in json_data:
                json_data[key] = tuple(json_data[key].strip('()').split(', '))
            cfg.CAM_TYPES = json_data

        # KEYFRAME_SELECTION
        cfg.INNOVATION_THRESH_PIX =int(config["KEYFRAME_SELECTION"]["INNOVATION_THRESH_PIX"])
        cfg.MIN_MATCHES =int(config["KEYFRAME_SELECTION"]["MIN_MATCHES"])
        cfg.RANSAC_THRESHOLD =int(config["KEYFRAME_SELECTION"]["ERROR_THRESHOLD"])
        cfg.RANSAC_ITERATIONS =int(config["KEYFRAME_SELECTION"]["MAX_ITERATIONS"])
        cfg.KFS_METHOD = config["KEYFRAME_SELECTION"]["METHOD"]
        cfg.KFS_LOCAL_FEATURE = config["KEYFRAME_SELECTION"]["LOCAL_FEATURE"]
        cfg.KFS_N_FEATURES = int(config["KEYFRAME_SELECTION"]["N_FEATURES"])
        cfg.ALIKE_MODEL = config["KEYFRAME_SELECTION"]["ALIKE_MODEL"]
        cfg.ALIKE_DEVICE = config["KEYFRAME_SELECTION"]["ALIKE_DEVICE"]
        cfg.ALIKE_SCORES_TH = float(config["KEYFRAME_SELECTION"]["ALIKE_SCORES_TH"])
        cfg.ALIKE_N_LIMIT = int(config["KEYFRAME_SELECTION"]["ALIKE_N_LIMIT"])
        cfg.ALIKE_SUBPIXEL = config["KEYFRAME_SELECTION"].getboolean("ALIKE_SUBPIXEL")
        cfg.ORB_SCALE_FACTOR = float(config["KEYFRAME_SELECTION"]["ORB_SCALE_FACTOR"])
        cfg.ORB_NLEVELS = int(config["KEYFRAME_SELECTION"]["ORB_NLEVELS"])
        cfg.ORB_EDGE_THRESHOLD = int(config["KEYFRAME_SELECTION"]["ORB_EDGE_THRESHOLD"])
        cfg.ORB_FIRST_LEVEL = int(config["KEYFRAME_SELECTION"]["ORB_FIRST_LEVEL"])
        cfg.ORB_WTA_K = int(config["KEYFRAME_SELECTION"]["ORB_WTA_K"])
        cfg.ORB_SCORE_TYPE = int(config["KEYFRAME_SELECTION"]["ORB_SCORE_TYPE"])
        cfg.ORB_PATCH_SIZE = int(config["KEYFRAME_SELECTION"]["ORB_PATCH_SIZE"])
        cfg.ORB_FAST_THRESHOLD = int(config["KEYFRAME_SELECTION"]["ORB_FAST_THRESHOLD"])

        # EXTERNAL_SENSORS
        cfg.USE_EXTERNAL_CAM_COORD = config["EXTERNAL_SENSORS"].getboolean("USE_EXTERNAL_CAM_COORD")
        cfg.CAMERA_COORDINATES_FILE = Path(config["EXTERNAL_SENSORS"]["CAMERA_COORDINATES_FILE"])

        # LOCAL_FEATURES
        cfg.LOCAL_FEAT_LOCAL_FEATURE = config["LOCAL_FEATURES"]["LOCAL_FEATURE"]
        cfg.LOCAL_FEAT_N_FEATURES = int(config["LOCAL_FEATURES"]["N_FEATURES"])
        cfg.SUPERGLUE_NMS_RADIUS = int(config["LOCAL_FEATURES"]["SUPERGLUE_NMS_RADIUS"])
        cfg.SUPERGLUE_KEYPOINT_THRESHOLD = float(config["LOCAL_FEATURES"]["SUPERGLUE_KEYPOINT_THRESHOLD"])
        cfg.SUPERGLUE_WEIGHTS = config["LOCAL_FEATURES"]["SUPERGLUE_WEIGHTS"]
        cfg.SUPERGLUE_SINKHORN_ITERATIONS = int(config["LOCAL_FEATURES"]["SUPERGLUE_SINKHORN_ITERATIONS"])
        cfg.SUPERGLUE_MATCH_THRESHOLD = float(config["LOCAL_FEATURES"]["SUPERGLUE_MATCH_THRESHOLD"])
        cfg.LOCAL_FEAT_ALIKE_MODEL = config["LOCAL_FEATURES"]["ALIKE_MODEL"]
        cfg.LOCAL_FEAT_ALIKE_DEVICE = config["LOCAL_FEATURES"]["ALIKE_DEVICE"]
        cfg.LOCAL_FEAT_ALIKE_SCORES_TH = float(config["LOCAL_FEATURES"]["ALIKE_SCORES_TH"])
        cfg.LOCAL_FEAT_ALIKE_N_LIMIT = int(config["LOCAL_FEATURES"]["ALIKE_N_LIMIT"])
        cfg.LOCAL_FEAT_ALIKE_SUBPIXEL = config["LOCAL_FEATURES"].getboolean("ALIKE_SUBPIXEL")
        cfg.LOCAL_FEAT_ORB_SCALE_FACTOR = float(config["LOCAL_FEATURES"]["ORB_SCALE_FACTOR"])
        cfg.LOCAL_FEAT_ORB_NLEVELS = int(config["LOCAL_FEATURES"]["ORB_NLEVELS"])
        cfg.LOCAL_FEAT_ORB_EDGE_THRESHOLD = int(config["LOCAL_FEATURES"]["ORB_EDGE_THRESHOLD"])
        cfg.LOCAL_FEAT_ORB_FIRST_LEVEL = int(config["LOCAL_FEATURES"]["ORB_FIRST_LEVEL"])
        cfg.LOCAL_FEAT_ORB_WTA_K = int(config["LOCAL_FEATURES"]["ORB_WTA_K"])
        cfg.LOCAL_FEAT_ORB_SCORE_TYPE = int(config["LOCAL_FEATURES"]["ORB_SCORE_TYPE"])
        cfg.LOCAL_FEAT_ORB_PATCH_SIZE = int(config["LOCAL_FEATURES"]["ORB_PATCH_SIZE"])
        cfg.LOCAL_FEAT_ORB_FAST_THRESHOLD = int(config["LOCAL_FEATURES"]["ORB_FAST_THRESHOLD"])
        cfg.LOCAL_FEAT_MIN_MATCHES = int(config["LOCAL_FEATURES"]["MIN_MATCHES"])

        # LOCAL_FEATURES
        cfg.LOCAL_FEAT2_USE_ADDITIONAL_FEATURES = config["LOCAL_FEATURES_2"].getboolean("USE_ADDITIONAL_FEATURES")
        cfg.LOCAL_FEAT2_N_FEATURES = int(config["LOCAL_FEATURES_2"]["N_FEATURES"])
        cfg.LOCAL_FEAT2_LOCAL_FEATURE = config["LOCAL_FEATURES_2"]["LOCAL_FEATURE"]

        # MATCHING
        cfg.KORNIA_MATCHER = config["MATCHING"]["KORNIA_MATCHER"]
        cfg.RATIO_THRESHOLD = float(config["MATCHING"]["RATIO_THRESHOLD"])
        cfg.GEOMETRIC_VERIFICATION = config["MATCHING"]["GEOMETRIC_VERIFICATION"]
        cfg.MAX_ERROR = float(config["MATCHING"]["MAX_ERROR"])
        cfg.CONFIDENCE = float(config["MATCHING"]["CONFIDENCE"])
        cfg.ITERATIONS = int(config["MATCHING"]["ITERATIONS"])
        cfg.LOOP_CLOSURE_DETECTION = config["MATCHING"].getboolean("LOOP_CLOSURE_DETECTION")
        cfg.VOCAB_TREE = config["MATCHING"]["VOCAB_TREE"]

        # INCREMENTAL_RECONSTRUCTION
        cfg.MIN_KEYFRAME_FOR_INITIALIZATION = int(config["INCREMENTAL_RECONSTRUCTION"]["MIN_KEYFRAME_FOR_INITIALIZATION"])

        self.cfg = cfg

        return self.cfg

    def get_colmap_path(self):
        """
        Get the path to the COLMAP executable based on the operating system

        Returns:
            str: The path to the COLMAP executable
        """
        OS = self.cfg.OS
        assert OS in [
            "windows",
            "linux",
        ], "Invalid OS. OS in conf.ini must be windows or linux"
        if OS == "windows":
            name = "COLMAP.bat"
        elif OS == "linux":
            name = "colmap"

        self.cfg.COLMAP_EXE_PATH = self.cfg.COLMAP_EXE_DIR / name

    def set_working_directory(self):
        """
        set_working_directory _summary_
        """
        self.cfg.CURRENT_DIR = Path(os.getcwd())
        self.cfg.TEMP_DIR = self.cfg.CURRENT_DIR / "temp"
        self.cfg.KEYFRAMES_DIR = self.cfg.CURRENT_DIR / "colmap_imgs"
        self.cfg.OUT_FOLDER = self.cfg.CURRENT_DIR / "outs"
        #self.cfg.DATABASE_DIR = self.cfg.CURRENT_DIR / "outs"

    def manage_output_folders(self) -> bool:
        """
        manage_output_folders _summary_

        Returns:
            bool: _description_
        """
        if self.cfg.TEMP_DIR.exists():
            shutil.rmtree(self.cfg.TEMP_DIR)
            shutil.rmtree(self.cfg.KEYFRAMES_DIR)
            shutil.rmtree(self.cfg.OUT_FOLDER)
            if os.path.exists("./keyframes.txt"):
                os.remove("./keyframes.txt")
        self.cfg.TEMP_DIR.mkdir()
        (self.cfg.TEMP_DIR / "pair").mkdir()
        self.cfg.KEYFRAMES_DIR.mkdir()
        self.cfg.OUT_FOLDER.mkdir()

        if self.cfg.IMGS_FROM_SERVER.exists():
            shutil.rmtree(self.cfg.IMGS_FROM_SERVER)
        for c in range(self.cfg.N_CAMERAS):
            camera_subfolder = self.cfg.IMGS_FROM_SERVER / f"cam{c}"
            camera_subfolder.mkdir(parents=True)

        if os.path.exists("./keyframes.pkl"):
            os.remove("./keyframes.pkl")

        if os.path.exists("./points3D.pkl"):
            os.remove("./points3D.pkl")

        return True
    
    def new_batch_solution(self):
        existing_batches = [int(i) for i in os.listdir(self.cfg.KEYFRAMES_DIR)]

        if bool(existing_batches) == False:
            current_dir = '0'
        else:
            current_dir = "{}".format(max(existing_batches) + 1)

        new_kf_dir = self.cfg.KEYFRAMES_DIR / current_dir
        new_kf_dir.mkdir()
        self.cfg.KF_DIR_BATCH = new_kf_dir
        new_out_dir = self.cfg.OUT_FOLDER / current_dir
        new_out_dir.mkdir()
        self.cfg.OUT_DIR_BATCH = new_out_dir
        self.cfg.DATABASE = self.cfg.OUT_FOLDER / current_dir / "db.db"

        return self.cfg

    def inizialize(self) -> edict:
        """
        inizialize _summary_

        Returns:
            edict: _description_
        """
        self.parse_config_file()
        self.set_working_directory()
        self.get_colmap_path()
        self.manage_output_folders()
        self.new_batch_solution()

        return self.cfg
