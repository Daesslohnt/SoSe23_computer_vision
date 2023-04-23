import numpy as np
import cv2

from Camera.Filter.filter import Filter


class ThresholdFilter(Filter):
    '''
    Applies a threshold function to the supplied image
    '''
    def apply_to(self, image: np.ndarray) -> np.ndarray:
        image = cv2.threshold(image, thresh=10, maxval=255, type=cv2.THRESH_BINARY)[1]

        return image
