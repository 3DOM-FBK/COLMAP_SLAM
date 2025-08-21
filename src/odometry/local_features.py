import os
import cv2
import time
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from src.thirdparty.ALIKED.nets.aliked import ALIKED
from transformers import AutoImageProcessor, SuperPointForKeypointDetection


class LocalFeatures:
    def __init__(
            self,
            image_width: int,
            image_height: int,
            config_local_features: dict,
            verbose: bool = False,
            ) -> None:
        self.verbose = verbose
        self.feature_name = config_local_features['features_name']
        self.image_width = image_width
        self.image_height = image_height
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.size ={
            "height": config_local_features['resize_height'],
            "width": config_local_features['resize_width'],
        }
        self.config_sp = config_local_features['superpoint']
        config_aliked = config_local_features['aliked']

        if self.feature_name == "superpoint":
            self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint", do_resize=self.config_sp['do_resize'], size=self.size)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cuda"):
                self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").cuda()
            else:
                self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
            self.model.eval()
        
        elif self.feature_name == "aliked":
            self.model = ALIKED(model_name=config_aliked['model_name'], device=self.device, top_k=config_aliked['top_k'], scores_th=config_aliked['scores_th'], n_limit=config_aliked['n_limit'])

    def superpoint(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        image = cv2.resize(image, (self.size['width'], self.size['height']))
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            images = [image]
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            scores = outputs.scores[0]
            topk = self.config_sp['top_k']
            if scores.shape[0] < topk:
                topk = scores.shape[0]
            topk_indices = torch.topk(scores, topk).indices

            kpts = outputs['keypoints'][0][topk_indices]
            kpts[:, 0] *= self.image_width / self.size['width']
            kpts[:, 1] *= self.image_height / self.size['height']
            keypoints[img_name] = kpts.to("cpu")
            descriptors[img_name] = outputs['descriptors'][0][topk_indices].to("cpu")
                
            del inputs, outputs
            torch.cuda.empty_cache()


        if self.verbose==True:
            t1 = time.time()
            print(f"Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors

    # Old version of superpoint() working in batches:
    def superpoint_batch(self, imgs_dir: Path, image_files: list, batch_size: int) -> Tuple[dict, dict]:
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        steps = len(image_files) // batch_size
        rest = len(image_files) % batch_size

        with torch.no_grad():
            for i in range(steps):
                images = []
                for k in range(batch_size):
                    img = image_files[i*batch_size+k]
                    image = Image.open(imgs_dir / img).convert("RGB")
                    images.append(image)
                inputs = self.processor(images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)

                for k in range(batch_size):
                    kpts = outputs['keypoints'][k]
                    kpts[:, 0] *= self.image_width / self.size['width']
                    kpts[:, 1] *= self.image_height / self.size['height']
                    keypoints[image_files[i*batch_size+k]] = kpts.to("cpu")
                    descriptors[image_files[i*batch_size+k]] = outputs['descriptors'][k].to("cpu")
                
                del inputs, outputs
                torch.cuda.empty_cache()

            if rest > 0:
                images = []
                for k in range(rest):
                    img = image_files[steps*batch_size+k]
                    image = Image.open(imgs_dir / img).convert("RGB")
                    images.append(image)

                inputs = self.processor(images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                for k in range(rest):
                    kpts = outputs['keypoints'][k]
                    kpts[:, 0] *= self.width / self.size['width']
                    kpts[:, 1] *= self.height / self.size['height']
                    keypoints[image_files[steps*batch_size+k]] = kpts.to("cpu")
                    descriptors[image_files[steps*batch_size+k]] = outputs['descriptors'][k].to("cpu")

                del inputs, outputs
                torch.cuda.empty_cache()

        if self.verbose==True:
            t1 = time.time()
            print(f"Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors

    def aliked(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        image = cv2.resize(image, (self.size['width'], self.size['height']))
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            pred = self.model.run(image)
            keypoints[img_name] = torch.from_numpy(pred['keypoints']).to("cpu")/resize_factor
            descriptors[img_name] = torch.from_numpy(pred['descriptors']).to("cpu")

        if self.verbose==True:
            t1 = time.time()
            print(f"Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors

    def extract(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        if self.feature_name == "superpoint":
            return self.superpoint(img_name, image)
        elif self.feature_name == "aliked":
            return self.aliked(img_name, image)
        