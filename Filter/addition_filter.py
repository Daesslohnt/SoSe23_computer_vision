import cv2
import numpy as np

from Filter.filter import Filter


class AdditionFilter(Filter):
    def __init__(self,weights:(int,int)=(0.5,0.5)):
        super().__init__()
        self.last_image = None
        self._weights = weights

    def apply_to(self,image:np.ndarray) -> np.ndarray:
        old_image = self.last_image
        if not type(old_image) is np.ndarray:
            old_image = image
        self.last_image = image

        cv2.addWeighted(old_image, self._weights[0], image, self._weights[1], 0.0, image)

        return image