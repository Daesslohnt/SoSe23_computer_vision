import math
import random
from time import sleep

import cv2

from Camera.webcam import Webcam


def phototool(save_path='./', photo_amount=10, before_photo_timer=5, between_photo_timer=0.5, tag="photo"):
    cam = Webcam()

    print("starting", tag, "in", before_photo_timer)
    sleep(before_photo_timer)
    print("starting:", tag)
    for photo_nr in range(photo_amount):
        name = str(photo_nr) + "_" + tag + "_" + str(math.floor(random.random() * 10_000)) + ".png"
        image = cam.get_image()
        cv2.imwrite(save_path + name, image)

        print(photo_amount-photo_nr)
        sleep(between_photo_timer)
    print("finished:", tag)


phototool(save_path="../daten/left_click/", photo_amount=50, tag="left_click")
phototool(save_path="../daten/double_click/", photo_amount=50, tag="double_click")
phototool(save_path="../daten/right_click/", photo_amount=50, tag="right_click")
phototool(save_path="../daten/scroll_up/", photo_amount=50, tag="scroll_up")
phototool(save_path="../daten/scroll_down/", photo_amount=50, tag="scroll_down")
phototool(save_path="../daten/hold_on/", photo_amount=50, tag="hold_on")
phototool(save_path="../daten/default/", photo_amount=50, tag="default")

'''
left_click - Daumen
double_click - Zeigerfinger
right_click - Mittelfinger 
scroll_up - Daumen hoch
scroll_down - Daumen runter
hold_on - Faust
default - offene Hand 
'''
