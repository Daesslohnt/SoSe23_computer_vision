import numpy as np

from Filter import Filter


class BWFilter(Filter):
    def apply_to(self,image:np.ndarray) -> np.ndarray:
        for row in range(len(image)):
            for col in range(len(image[row])):
                avg_brightness = sum(image[row][col])/len(image[row][col])
                image[row][col] = [avg_brightness,avg_brightness,avg_brightness]
        return image