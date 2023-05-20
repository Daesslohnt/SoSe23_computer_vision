import cv2
import numpy as np

from Camera.Filter.filter import Filter


class BlurFilter(Filter):
    '''
    Blurs the supplied image using a bilateral filter.
    '''

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, 9, 75, 75)
