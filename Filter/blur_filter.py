import cv2
import numpy as np

from Filter.filter import Filter


class BlurFilter(Filter):
    def apply_to(self,image:np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image,9,75,75)