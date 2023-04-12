import numpy as np
from cv2 import resize

from Filter.filter import Filter


class ResizeFilter(Filter):
    def __init__(self, size=None):
        super().__init__()
        if size is None:
            size = [128, 128]
        self.size = size

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        resize(image, self.size, image)

        return image
