import numpy as np
import cv2
from copy import deepcopy
import logging
import importlib


def make_match_plot(
    img: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray
) -> np.ndarray:
    """
    Generates a visualization of the matched points.
    Args:
        img (numpy.ndarray): Current image.
        mpts1 (numpy.ndarray): Matched points from the previous frame.
        mpts2 (numpy.ndarray): Matched points from the current frame.
    Returns:
        numpy.ndarray: An image showing the matched points.
    """
    match_img = deepcopy(img)
    for pt1, pt2 in zip(mpts1, mpts2):
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
        cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)

    return match_img


class Matcher:
    def __init__(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ):
        self.desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
        self.desc2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)

    def mnn_matcher_cosine(self) -> np.ndarray:
        """
        Computes the nearest neighbor matches between two sets of descriptors.
        Args:
            desc1 (numpy.ndarray): First set of descriptors.
            desc2 (numpy.ndarray): Second set of descriptors.
        Returns:
            numpy.ndarray: An array of indices indicating the nearest neighbor matches between the two sets of descriptors.
        """
        sim = self.desc1 @ self.desc2.transpose()
        sim[sim < 0.8] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = ids1 == nn21[nn12]
        matches = np.stack([ids1[mask], nn12[mask]])
        # matches = np.stack([ids1, nn12])

        return matches.transpose()

    def make_plot(
        self, img: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray
    ) -> np.ndarray:
        """
        Generates a visualization of the matched points.
        Args:
            img (numpy.ndarray): Current image.
            mpts1 (numpy.ndarray): Matched points from the previous frame.
            mpts2 (numpy.ndarray): Matched points from the current frame.
        Returns:
            numpy.ndarray: An image showing the matched points.
        """

        return make_match_plot(img, mpts1, mpts2)

    def geometric_verification(
        self,
        threshold: float = 1,
        confidence: float = 0.9999,
        max_iters: int = 10000,
        laf_consistensy_coef: float = -1.0,
        error_type: str = "sampson",
        symmetric_error_check: bool = True,
        enable_degeneracy_check: bool = True,
    ) -> np.ndarray:
        """
        Computes the fundamental matrix and inliers between the two images using geometric verification.

        Args:
            threshold (float): Pixel error threshold for considering a correspondence an inlier.
            confidence (float): The required confidence level in the results.
            max_iters (int): The maximum number of iterations for estimating the fundamental matrix.
            laf_consistensy_coef (float): The weight given to Local Affine Frame (LAF) consistency term for pydegensac.
            error_type (str): The error function used for computing the residuals in the RANSAC loop.
            symmetric_error_check (bool): If True, performs an additional check on the residuals in the opposite direction.
            enable_degeneracy_check (bool): If True, enables the check for degeneracy using SVD.

        Returns:
            np.ndarray: A Boolean array that masks the correspondences that were identified as inliers.

        TODO: allow parameters for both MAGASAC++ and pydegensac to be passed in (currentely only pydegensac is supported).
        TODO: add support for other geometric verification methods.
        """

        try:
            pydegensac = importlib.import_module("pydegensac")
            use_pydegensac = True
        except:
            logging.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            use_pydegensac = False
        try:
            if use_pydegensac:
                _, mask = pydegensac.findFundamentalMatrix(
                    self.mpts1,
                    self.mpts2,
                    px_th=threshold,
                    conf=confidence,
                    max_iters=max_iters,
                    laf_consistensy_coef=laf_consistensy_coef,
                    error_type=error_type,
                    symmetric_error_check=symmetric_error_check,
                    enable_degeneracy_check=enable_degeneracy_check,
                )
                logging.info(f"Pydegensac found {mask.sum()}/{len(mask)} inliers")
            else:
                _, inliers = cv2.findFundamentalMat(
                    self.mpts1,
                    self.mpts2,
                    cv2.USAC_MAGSAC,
                    0.5,
                    0.999,
                    100000,
                )
                mask = inliers > 0
                logging.info(f"MAGSAC++ found {mask.sum()}/{len(mask)}")
            self.mpts1 = self.mpts1[mask, :]
            self.mpts2 = self.mpts2[mask, :]
        except ValueError as err:
            logging.error(f"Unable to perform geometric verification: {err}.")

        return self.inlMask
