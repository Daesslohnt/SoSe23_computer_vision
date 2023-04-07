# program to capture single image from webcam in python
import sys
import time
from time import *

import numpy
import numpy as np
# importing OpenCV

from cv2.cv2 import *

if __name__ == "__main__":

    # initialize the camera
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = VideoCapture(cam_port)

    act0 = 0
    act1 = 0

    image0 = cvtColor(cam.read()[1], COLOR_BGR2GRAY)
    old_image = image = image0
    height = len(image0)
    width = len(image0[0])

    while True:
        # reading the input using the camera
        im_rgb = cam.read()[1]
        image1 = cvtColor(im_rgb, COLOR_BGR2GRAY)

        # image = GaussianBlur(image, (0, 0), sigmaX=2, sigmaY=2)
        image1 = GaussianBlur(image1, (0, 0), sigmaX=5, sigmaY=5)

        image = absdiff(image1, image0)

        # image = GaussianBlur(image, (0, 0), sigmaX=15, sigmaY=15)

        kernel = np.ones((5, 5))
        image = dilate(image, kernel, 1)
        image = threshold(src=image, thresh=10, maxval=255, type=THRESH_BINARY)[1]
        # contours,  = findContours(image=image, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE)

        # drawContours(image=im_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=LINE_AA)

        image0 = image1

        # image = Canny(image, 100, 200, None, 3, False)

        addWeighted(old_image, 0.7, image, 1.0, 0.0, image)

        imshow("GeeksForGeeks", image)
        old_image = image
        imshow("rgb", im_rgb)
        # print(len(image[0]))
        # saving image in local storage
        # imwrite("GeeksForGeeks.png", image)
1
        # If keyboard interrupt occurs, destroy image
        # window
        act1 = perf_counter()
        print(act1 - act0)
        act0 = act1
        waitKey(1)

        # if cv2.waitKey(1) == ord('q'):
        #     break

        # destroyWindow("GeeksForGeeks")
