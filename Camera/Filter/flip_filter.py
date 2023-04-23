import cv2
import numpy as np

from Camera.Filter.filter import Filter

class FlipFilter(Filter):
    def apply_to(self, image: np.ndarray) -> np.ndarray:

        return cv2.flip(image, 1)
