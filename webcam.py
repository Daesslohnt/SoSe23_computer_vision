import time

from cv2.cv2 import *

from Filter.filter import Filter


class Webcam:

    def __init__(self, cam_port=0):
        self.pipelines = {
            "default": [Filter()]
        }
        self.cam = VideoCapture(cam_port)

        if not self.cam.isOpened():
            raise EnvironmentError("Camera could not be opened(no camera?). Exiting ...")

    def add_pipeline(self, name: str, pipeline: [Filter]) -> None:
        self.pipelines[name] = pipeline

    def get_image(self, pipeline_name: str = "default"):
        image = self.cam.read()[1]
        #        read_success = True
        #        if not read_success:
        #            EnvironmentError("Can't receive frame (stream end?). Exiting ...")

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

    def save_image(self, image_name: str = time.time(), pipeline_name: str = "default"):
        cv2.imwrite(image_name, self.get_image(pipeline_name))
