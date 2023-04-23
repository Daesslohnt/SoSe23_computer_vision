import numpy as np
import cv2

from Camera.Filter.filter import Filter


class GreyFilter(Filter):
    '''
    Converts the supplied image to grayscale.
    '''
    def apply_to(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
