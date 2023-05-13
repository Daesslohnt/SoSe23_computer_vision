import cv2
import numpy as np

from Camera.Filter.filter import Filter


class ResizeFilter(Filter):
    '''
    Resizes the supplied image to a given size. Default is 128*128.
    '''

    def __init__(self, size=None):
        super().__init__()
        if size is None:
            size = [128, 128]
        self._size = size

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        cv2.resize(image, self._size, image)

        return image
