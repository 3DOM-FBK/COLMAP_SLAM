config = {
    "general": {
        "cfg_file": "config.ini",
        "log_level": "DEBUG"
    },

    "frames_reader": {
        "start": 0,
        "end": 100000000,
        "show_camera_flow": True,
        "path_to_frames_dir": "C:/Users/threedom/Desktop/3dom-github/COLMAP_SLAM/prova",
        "step": 1,
        "sleep": 0.25,
        "equalize": False,
    },

    "keyframe_selector": {
        "show": True,
        "max_keypoints": 2000,
    },

    "calibration": {
        "n_camera": 1,
    }
}