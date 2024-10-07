import sys
from copy import copy
import cv2
import numpy as np
import torch
from torch import nn
from pathlib import Path
from ..thirdparty.SuperGluePretrainedNetwork.models import superpoint
from typing import Optional, Tuple, TypedDict, Union

# TODO: Use Superpoint implementation from LightGlue

# The original keypoint sampling is incorrect. We patch it here but
# we don't fix it upstream to not impact exisiting evaluations.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    _default_conf = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": True,
    }
    required_inputs = ["image"]
    detection_noise = 2.0

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self._default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    def _init(self, conf):
        if conf["fix_sampling"]:
            superpoint.sample_descriptors = sample_descriptors_fix_sampling
        self.net = superpoint.SuperPoint(conf)

    def _forward(self, data):
        return self.net(data)


class SuperPointExtractor():

    _default_conf = {
        "name": "superpoint",
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": False,
    }
    required_inputs = ["image"]
    grayscale = True
    descriptor_size = 256
    detection_noise = 2.0
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config: dict):
        self._extractor = SuperPoint(config).eval().to(self._device)

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image using the SuperPoint model.

        Args:
            image (np.ndarray): The input image as a numpy array.

        Returns:
            np.ndarray: A dictionary containing the extracted features. The keys represent different feature types, and the values are numpy arrays.

        """
        image_ = self._frame2tensor(image, self._device)
        feats = self._extractor({"image": image_})
        feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
        feats = {k: v.cpu().numpy() for k, v in feats.items()}
        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

    def extract(self, img: Union[Path, str]) -> np.ndarray:
        """
        Extract features from an image. This is the main method of the feature extractor.

        Args:
                img: Image to extract features from. It can be either a path to an image or an Image object

        Returns:
                List of features extracted from the image. Each feature is a 2D NumPy array
        """

        if isinstance(img, str):
            im_path = Path(img)
            image = cv2.imread(str(im_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(img, Path):
            im_path = img
            image = cv2.imread(str(im_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(img, np.ndarray):
            image = img
        else:
            raise TypeError("Invalid image path. 'img' must be a string, a Path or an Image object")

        features = self._extract(image)

        return features
