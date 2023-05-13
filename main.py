import os.path
import time

import cv2

from Camera.Filter.abs_diff_filter import AbsDiffFilter
from Camera.Filter.addition_filter import AdditionFilter
from Camera.Filter.blur_filter import BlurFilter
from Camera.Filter.dilate_filter import DilateFilter
from Camera.Filter.filter import *
from Camera.Filter.flip_filter import FlipFilter
from Camera.Filter.grey_filter import GreyFilter
from Camera.Filter.threshold_filter import ThresholdFilter
from Camera.webcam import Webcam

cam = Webcam()
cam.add_pipeline("Filter", [GreyFilter(), BlurFilter(), AbsDiffFilter(), ThresholdFilter(), DilateFilter(np.ones(1)),
                            AdditionFilter((0.9, 1.0)), FlipFilter()])
path = os.path.abspath("/home/benedikts/PycharmProjects/computervisionss23/Hello.png")
old_time = 0

while True:

    image = cam.get_image()
    image2 = cam.get_image("Filter")

    cv2.imshow("Webcam", image)
    cv2.imshow("Filter", image2)

    if cv2.waitKey(1) == ord('q'):
        break

    new_time = time.perf_counter()
    res = time.perf_counter() - old_time
    old_time = new_time
    print(res)
