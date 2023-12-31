import cv2
import numpy as np

from Camera.Filter.filter import Filter


class AbsDiffFilter(Filter):
    '''
    Calculates the difference of the last and current supplied images.
    '''

    def __init__(self):
        super().__init__()
        self.last_image = None

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        old_image = self.last_image
        if not type(old_image) is np.ndarray:
            old_image = image
        self.last_image = image

        image = cv2.absdiff(image, old_image)

        return image
