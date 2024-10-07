import os
import cv2
import sys
import time
import torch
import numpy as np

from torch import nn
from copy import copy
from pathlib import Path
from .extractors.superpoint import SuperPointExtractor


class KeyframeSelector:
    def __init__(
            self,
            frames_dir: Path,
            keyframes_dir: Path,
            show: bool = True,
            n_camera: int = 1,
            max_keypoints: int = 1000,
            ):
        
        first_img_path = frames_dir / 'cam0' / os.listdir(frames_dir / 'cam0')[0]
        shape = cv2.imread(str(first_img_path)).shape
        hight, width = shape[0], shape[1]
        self.clock = 0.001
        self.last_keyframe = 0
        self.candidate_keyframe = 1
        self.show = show
        self.frames_dir= frames_dir
        self.keyframes_dir = keyframes_dir
        self.n_camera = n_camera
        self.max_keypoints = max_keypoints

        self.last_frame = {
            "id": 0,
            "cam0": {
                "keypoints":   None,
                "descriptors": None,
                "img_gray": None,
            },
            "cam1": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam2": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam3": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam4": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
        }

        self.new_frame = {
            "id": 1,
            "cam0": {
                "keypoints":   None,
                "descriptors": None,
                "img_gray": None,
            },
            "cam1": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam2": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam3": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
            "cam4": {
                "keypoints":   np.empty((max_keypoints, 2), dtype=np.float32),
                "descriptors": np.empty((max_keypoints, 256), dtype=np.float32),
                "img_gray": np.empty((hight, width), dtype=np.uint8),
            },
        }

        conf_SP = {
            "name": "superpoint",
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": self.max_keypoints,
            "remove_borders": 4,
            "fix_sampling": False,
        }

        self._extractor = SuperPointExtractor(conf_SP)

        self.lk_params = dict(
                            winSize=(15, 15),        # Window size for the flow search
                            maxLevel=2,              # Maximum pyramid levels
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                            )

    def _initialize(self):
        while True:
            jpg_files = [f for f in os.listdir(self.frames_dir / 'cam0') if f.endswith('.jpg')]
            if len(jpg_files) >= 2:
                for camera in range(self.n_camera):
                    img_path = self.frames_dir / f'cam{camera}' /  f"{0:06d}.jpg"
                    image = cv2.imread(str(img_path))
                    self.last_frame[f'cam{camera}']['img_gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    feats = self._extractor.extract(self.last_frame[f'cam{camera}']['img_gray'])
                    self.last_frame[f'cam{camera}']['keypoints'] = feats['keypoints']
                    self.last_frame[f'cam{camera}']['descriptors'] = feats['descriptors']
                break
            time.sleep(self.clock)
    
    def _track(self, camera: str = 'cam0'):
        #img = self.last_frame[camera]['img_gray']
        #p = self.last_frame[camera]['keypoints']
        #for i in range(p.shape[0]):
        #    x, y = int(p[i, 0]), int(p[i, 1])
        #    #img = cv2.line(img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)  # Draw line for flow
        #    #img = cv2.circle(img, (int(a), int(b)), 2, (0, 0, 255), -1) 
        #    img = cv2.circle(img, (x,y), 2, (0, 0, 255), -1) 
        #cv2.imshow('Keyframes', img)
        #cv2.waitKey(1)
        #quit()
        print('kanade')
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame[camera]['img_gray'], self.new_frame[camera]['img_gray'], self.last_frame[camera]['keypoints'], None, **self.lk_params)
        print('finished')
        #print(self.last_frame[camera]['keypoints'])
        #print(p1);quit()
        self.new_frame[camera]['keypoints'] = p1[np.squeeze(st)==1, :]
        valid_indexes = np.where(st == 1)[0]
        self.matches = np.hstack((valid_indexes.reshape(-1, 1), np.arange(0,self.new_frame[camera]['keypoints'].shape[0]).reshape(-1, 1)))
        good_old = self.last_frame[camera]['keypoints']#[valid_indexes, :]
        good_new = self.new_frame[camera]['keypoints']
        return p1, self.last_frame[camera]['keypoints']

    def run(self):
        self._initialize()
        while True:
            for camera in range(self.n_camera):
                last_frame_path = self.frames_dir / f'cam{camera}' /  f"{self.last_frame['id']:06d}.jpg"
                new_frame_path = self.frames_dir / f'cam{camera}' / f"{self.new_frame['id']:06d}.jpg"
                if last_frame_path.exists() and new_frame_path.exists():
                    img_old = cv2.imread(str(last_frame_path))
                    self.last_frame['img_gray'] = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
                    img_new = cv2.imread(str(new_frame_path))
                    self.new_frame['img_gray'] = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
                    good_new, good_old = self._track(f"cam{camera}")

                    if self.show and camera == 0:
                        for i in range(good_new.shape[0]):
                            x, y = int(good_new[i, 0]), int(good_new[i, 1])
                            #img = cv2.line(img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)  # Draw line for flow
                            #img = cv2.circle(img, (int(a), int(b)), 2, (0, 0, 255), -1) 
                            img_new = cv2.circle(img_new, (x,y), 2, (0, 0, 255), -1) 
                        cv2.imshow('Keyframes', img_new)
                        cv2.waitKey(10000)
                        quit()


                    self.candidate_keyframe += 1
            time.sleep(self.clock)