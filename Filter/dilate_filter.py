import numpy
import numpy as np
from cv2 import dilate

from Filter.filter import Filter


class DilateFilter(Filter):
    def __init__(self, kernel: numpy.array(1) = np.ones((5, 5))):
        super().__init__()
        self.kernel = kernel

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        image = dilate(image, self.kernel, 1)

        return image
