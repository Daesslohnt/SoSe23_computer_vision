import cv2
import numpy as np

from Filter import Filter


class BlurFilter(Filter):
    def apply_to(self,image:np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image,(3,3),0)