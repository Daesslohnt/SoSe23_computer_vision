import numpy as np
from cv2 import threshold, THRESH_BINARY

from Filter.filter import Filter


class ThresholdFilter(Filter):
    '''
    Applies a threshold function to the supplied image
    '''
    def apply_to(self, image: np.ndarray) -> np.ndarray:
        image = threshold(image, thresh=10, maxval=255, type=THRESH_BINARY)[1]

        return image
