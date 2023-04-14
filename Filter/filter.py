import numpy as np


class Filter:
    '''
    This class serves as the parent class for all filters used in the Webcam class
    Filters provide their functionality in the apply_to() method.
    '''

    def __init__(self):
        pass

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        '''
        applies an effect to the supplied image

        :param image: numpy.ndarray: image to be filtered
        :return: numpy.ndarray: filtered image
        '''
        return image
