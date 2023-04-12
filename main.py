import time

from cv2 import cv2

from Filter.abs_diff_filter import AbsDiffFilter
from Filter.addition_filter import AdditionFilter
from Filter.blur_filter import BlurFilter
from Filter.dilate_filter import DilateFilter
from Filter.filter import *
from Filter.grey_filter import GreyFilter
from Filter.threshold_filter import ThresholdFilter
from webcam import Webcam

cam = Webcam()
cam.add_pipeline("Filter", [GreyFilter(), BlurFilter(), AbsDiffFilter(), ThresholdFilter(), DilateFilter(np.ones(1)),
                            AdditionFilter((0.5, 1.0))])

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
