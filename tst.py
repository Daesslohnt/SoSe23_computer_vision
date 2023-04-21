# program to capture single image from webcam in python
import builtins
import math
import sys
import time
from pynput.mouse import *
from time import *

import numpy as np

from cv2.cv2 import *

if __name__ == "__main__":

    # initialize the camera
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = VideoCapture(cam_port)

    mouse = Controller()

    print(cam.get(CAP_PROP_FRAME_WIDTH), cam.get(CAP_PROP_FRAME_HEIGHT))

    act0 = 0
    act1 = 0
    act2 = 0

    xfin = 300
    yfin = 200

    xtop = 0
    ytop = 0
    wtop = 0
    htop = 0

    xfinal = 0
    yfinal = 0

    top = (0, 0)

    image0 = cvtColor(cam.read()[1], COLOR_BGR2GRAY)
    old_image = image = image0
    height = len(image0)
    width = len(image0[0])
    white = np.zeros([480, 640, 3], dtype=np.uint8)
    white[:, :] = [255, 255, 255]

    while True:
        act0 = perf_counter()
        # reading the input using the camera
        im_rgb = flip(cam.read()[1], 1)

        image1 = cvtColor(im_rgb, COLOR_BGR2GRAY)

        gray = im_rgb.copy()
        gray[:, :320] = 0
        gray[240:, :] = 0
        blue, green, red = split(gray)
        M = np.maximum(np.maximum(red, green), blue)
        red[red < M] = 0
        green[green < M] = 0
        blue[blue < M] = 0
        blue = add(blue, 50)
        green = add(green, 50)
        red[red < blue] = 0
        red[red < green] = 0
        red = threshold(src=red, thresh=50, maxval=255, type=THRESH_BINARY)[1]
        kernel = np.ones((10, 10))
        red = erode(red, kernel, 1)
        red = dilate(red, kernel, 1)
        contours, _ = findContours(image=red, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE)
        if len(contours):
            hulls = []
            for contour in contours:
                hulls.append(convexHull(contour))
            drawContours(image=im_rgb, contours=hulls, contourIdx=-1, color=(255, 0, 0), thickness=3, lineType=LINE_AA)
            hulls.sort(key=contourArea, reverse=True)
            hull = hulls[0].tolist()
            hull = [k[0] for k in hull]
            hull.sort(key=lambda l: l[1])
            top = hull[0]
            im_rgb = circle(img=im_rgb, center=top, radius=10, color=(255, 0, 0), thickness=3)
            top[0] = int((top[0] - 320))
            top[1] = int((top[1]))
        imshow("red", red)

        gray = im_rgb.copy()
        gray = cvtColor(gray, COLOR_BGR2GRAY)
        gray = GaussianBlur(gray, (0, 0), sigmaX=3, sigmaY=3)
        gray = threshold(src=gray, thresh=240, maxval=255, type=THRESH_BINARY)[1]
        # kernel = np.ones((15, 15))
        # gray = erode(gray, kernel, 1)
        contours, _ = findContours(image=gray, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE)
        # drawContours(image=im_rgb, contours=contours, contourIdx=-1, color=125, thickness=3, lineType=LINE_AA)
        # gray = gray[builtins.max([0, int(yfin - 100)]): int(yfin + 100), builtins.max([0, int(xfin - 100)]): int(xfin + 100)]

        distmin = 1000
        xmin = 0
        ymin = 0
        wmin = 0
        hmin = 0
        if len(contours):
            for contour in contours:
                x, y, w, h = boundingRect(contour)
                # im_rgb = rectangle(im_rgb, (int(x), y), (x + w, y + h), (0, 255, 0), 5)
                if 200 < w * h:  # and (w > 20) and (h > 20) and (0.7 < w / h < 1.4) and (np.sum(gray[y:(y + h), x: (x + h)] == 255) / (w * h) > 0.5):
                    dist = math.dist([x, y], [xfin, yfin])
                    if dist < distmin:
                        distmin = dist
                        xmin = x
                        ymin = y
                        wmin = w
                        hmin = h
        imshow("gray", gray)

        # image = GaussianBlur(image, (0, 0), sigmaX=5, sigmaY=5)

        image1 = GaussianBlur(image1, (0, 0), sigmaX=5, sigmaY=5)
        image = absdiff(image1, image0)

        # image = GaussianBlur(image, (0, 0), sigmaX=15, sigmaY=15)

        kernel = np.ones((5, 5))
        image = dilate(image, kernel, 3)
        image = threshold(src=image, thresh=10, maxval=255, type=THRESH_BINARY)[1]
        # image = adaptiveThreshold(src=image, maxValue=255, adaptiveMethod=ADAPTIVE_THRESH_MEAN_C, thresholdType=THRESH_BINARY, blockSize=15, C=5)
        contours, _ = findContours(image=image, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE)
        drawContours(image=image, contours=contours, contourIdx=-1, color=125, thickness=1, lineType=LINE_AA)
        if len(contours):
            sizesum = 1
            xsum = 0
            ysum = 0
            ytop = 480
            for contour in contours:
                x, y, w, h = boundingRect(contour)
                size = w * h
                sizesum += size
                xsum += x * size
                ysum += y * size
                if y < ytop:
                    ytop = y
                xtop = x
                wtop = w
                htop = h
        # im_rgb = rectangle(im_rgb, (int(x + 0.5 * w), y), (x + 5, y + 5), (0, 255, 0), 5)
        ym = int(ysum / sizesum)
        xm = int(xsum / sizesum)
        xtop2 = int(xtop + 0.5 * wtop)
        tp = 0.1
        tp2 = 0.3
        xfin = xfin * (1 - tp) + xtop2 * tp
        yfin = yfin * (1 - tp) + ytop * tp
        xfinal = xfinal * (1 - tp2) + top[0] * tp2
        yfinal = yfinal * (1 - tp2) + top[1] * tp2

        # im_rgb = rectangle(im_rgb, (int(xfin), int(yfin)), (int(xfin) + 2, int(yfin) + 2), (0, 255, 0), 5)
        # im_rgb = rectangle(im_rgb, (int(xfinal), int(yfinal)), (int(xfinal) + 20, int(yfinal) + 20), (0, 0, 255), 5)
        mouse.position = (int(xfinal * 6), int(yfinal * 4.5))


        image0 = image1

        # image = Canny(image, 100, 200, None, 3, False)

        # addWeighted(old_image, 0.7, image, 1.0, 0.0, image)

        imshow("GeeksForGeeks", image)
        old_image = image
        imshow("rgb", im_rgb)
        # print(len(image[0]))
        # saving image in local storage
        # imwrite("GeeksForGeeks.png", image)

        # If keyboard interrupt occurs, destroy image
        # window
        act1 = perf_counter()
        diff = act1 - act0
        # print("diff= " + str(diff))

        if waitKey(1) == ord('q'):
            break
    act2 = perf_counter()
    print("ttal= " + str(act2 - act0))

destroyWindow("GeeksForGeeks")
