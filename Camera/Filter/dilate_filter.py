import cv2
import numpy
import numpy as np

from Camera.Filter.filter import Filter


class DilateFilter(Filter):
    '''
    Dilates the supplied image with a given kernel. Default kernel size is 5*5.
    '''

    def __init__(self, kernel: numpy.array(1) = np.ones((5, 5))):
        super().__init__()
        self._kernel = kernel

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        image = cv2.dilate(image, self._kernel, 1)

        return image
