import numpy as np
from cv2 import cv2

from Filter.filter import Filter


class GreyFilter(Filter):
    def apply_to(self,image:np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
