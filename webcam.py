import time

import numpy as np
from cv2.cv2 import *

from Filter.filter import Filter


class Webcam:
    '''
    Webcam class used for getting images from the Webcam and applying filters to it.
    '''

    def __init__(self, cam_port:int=0):
        '''
        Creates a new Webcam used for getting images from the Webcam and applying filters to it.

        :param cam_port: int: your camera, default is 0
        '''
        self.pipelines = {
            "default": [Filter()]
        }
        self.cam = VideoCapture(cam_port)

        if not self.cam.isOpened():
            raise EnvironmentError("Camera could not be opened. Exiting ...")

    def add_pipeline(self, name: str, pipeline: [Filter]) -> None:
        '''
        Add a new pipeline to the Webcam. The name supplied is used in the get_image() method to address this pipeline.

        :param name: str: Name of the pipeline
        :param pipeline: [Filter]: List of filters to be used in this pipeline
        :return: None
        '''
        self.pipelines[name] = pipeline


    def get_image(self, pipeline_name: str = "default") -> np.ndarray:
        '''
        Gets an image from the webcam and applies the specified pipeline to it.
        The build-in "default" pipeline supplies the original picture.

        :param pipeline_name: str: pipeline to be applied, default is "default"
        :return: numpy.ndarray: the filtered image
        '''
        image = self.cam.read()[1]

        pipeline = self._get_pipeline(pipeline_name)

        return self._apply_pipeline(image, pipeline)

    def _get_pipeline(self, pipeline_name: str) -> [Filter]:
        pipeline = self.pipelines[pipeline_name]
        if not pipeline:
            pipeline = self.pipelines["default"]

        return pipeline

    @staticmethod
    def _apply_pipeline(image, pipeline: [Filter]):
        for filter_instance in pipeline:
            image = filter_instance.apply_to(image)

        return image

    def save_image(self, image_name: str = time.time(), pipeline_name: str = "default") -> None:
        '''
        Same as get_image, but instead of returning, the image is saved to the specified name.

        :param image_name: the name of the saved image
        :param pipeline_name: str: pipeline to be applied, default is "default"
        :return: None
        '''
        cv2.imwrite(image_name, self.get_image(pipeline_name))

Filter