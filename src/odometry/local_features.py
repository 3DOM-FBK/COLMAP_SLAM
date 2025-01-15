import os
import cv2
import torch

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
            ) -> None:
        self.feature_name = config_local_features['features_name']
        self.image_width = image_width
        self.image_height = image_height
        config_sp = config_local_features['superpoint']
        config_aliked = config_local_features['aliked']

        if self.feature_name == "superpoint":
            self.size ={
                "height": config_sp['resize_height'],
                "width": config_sp['resize_width'],
            }
            self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint", do_resize=config_sp['do_resize'], size=self.size)
            self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        
        elif self.feature_name == "aliked":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ALIKED(model_name=config_aliked['model_name'], device=self.device, top_k=config_aliked['top_k'], scores_th=config_aliked['scores_th'], n_limit=config_aliked['n_limit'])
        
    def superpoint(self, imgs_dir: Path, image_files: list, batch_size: int) -> Tuple[dict, dict]:
        keypoints = {}
        descriptors = {}
        steps = len(image_files) // batch_size
        rest = len(image_files) % batch_size

        with torch.no_grad():
            for i in tqdm(range(steps)):
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
                    keypoints[image_files[i*batch_size+k]] = kpts
                    descriptors[image_files[i*batch_size+k]] = outputs['descriptors'][k]
                
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
                    keypoints[image_files[steps*batch_size+k]] = kpts
                    descriptors[image_files[steps*batch_size+k]] = outputs['descriptors'][k]

                del inputs, outputs
                torch.cuda.empty_cache()

        return keypoints, descriptors

    def aliked(self, imgs_dir: Path, image_files: list, batch_size: int) -> Tuple[dict, dict]:
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            for img in tqdm(image_files):
                cv2_img = cv2.imread(str(imgs_dir / img))
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                pred = self.model.run(img_rgb)
                keypoints[img] = pred['keypoints']
                descriptors[img] = pred['descriptors']

        return keypoints, descriptors

    def extract(self, imgs_dir: Path, image_files: list, batch_size: int) -> Tuple[dict, dict]:
        if self.feature_name == "superpoint":
            return self.superpoint(imgs_dir, image_files, batch_size)
        elif self.feature_name == "aliked":
            return self.aliked(imgs_dir, image_files, batch_size)
        