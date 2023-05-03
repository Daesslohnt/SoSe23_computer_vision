from enum import Enum

import cv2
import numpy as np

from Helper.rectangle import Rectangle


class Hand:

    def __init__(self, landmark, handedness):
        self.landmark = landmark
        self.handedness = handedness

    def get_bounding_box(self, hand_region):
        if hand_region is HandRegion.PALM:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in [0, 5, 9, 13, 17]], [640, 480])
        elif hand_region is HandRegion.THUMB:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in range(2, 5)], [640, 480])
        elif hand_region is HandRegion.INDEX:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in range(6, 9)], [640, 480])
        elif hand_region is HandRegion.MIDDLE:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in range(10, 13)], [640, 480])
        elif hand_region is HandRegion.RING:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in range(14, 17)], [640, 480])
        elif hand_region is HandRegion.PINKY:
            return Rectangle([(self.landmark[i].x, self.landmark[i].y) for i in range(18, 21)], [640, 480])
        else:
            return None

    def is_extended(self, finger):
        hand_vec = [self.landmark[9].x - self.landmark[0].x,
                    self.landmark[9].y - self.landmark[0].y]
        if finger is HandRegion.THUMB:
            thumb_vec = [self.landmark[4].x - self.landmark[2].x,
                         self.landmark[4].y - self.landmark[2].y]
            return self.landmark[2].x > self.landmark[4].x
        elif finger is HandRegion.INDEX:
            index_vec = [self.landmark[8].x - self.landmark[6].x,
                         self.landmark[8].y - self.landmark[6].y]
            return dot(hand_vec, index_vec) > 0
        elif finger is HandRegion.MIDDLE:
            middle_vec = [self.landmark[12].x - self.landmark[10].x,
                          self.landmark[12].y - self.landmark[10].y]
            return self.landmark[10].y > self.landmark[12].y
        elif finger is HandRegion.RING:
            ring_vec = [self.landmark[16].x - self.landmark[14].x,
                        self.landmark[16].y - self.landmark[14].y]
            return self.landmark[14].y > self.landmark[16].y
        elif finger is HandRegion.PINKY:
            pinky_vec = [self.landmark[20].x - self.landmark[18].x,
                         self.landmark[20].y - self.landmark[18].y]
            return self.landmark[18].y > self.landmark[20].y
        else:
            return False

    def draw_vecs(self, image, dims):
        draw_arrow = lambda vec, pnt: cv2.arrowedLine(image, [int(c) for c in pnt], [int(c) for c in pnt + vec],
                                                     (0, 0, 255), 2) if vec.all() and pnt.all() else None

        hand_vec = [self.landmark[9].x - self.landmark[0].x,
                    self.landmark[9].y - self.landmark[0].y] * dims
        hand_pnt = [self.landmark[0].x, self.landmark[0].y] * dims

        thumb_vec = [self.landmark[4].x - self.landmark[2].x,
                     self.landmark[4].y - self.landmark[2].y] * dims
        thumb_pnt = [self.landmark[2].x, self.landmark[2].y] * dims

        index_vec = [self.landmark[8].x - self.landmark[6].x,
                     self.landmark[8].y - self.landmark[6].y] * dims
        index_pnt = [self.landmark[6].x, self.landmark[6].y] * dims

        middle_vec = [self.landmark[12].x - self.landmark[10].x,
                      self.landmark[12].y - self.landmark[10].y] * dims
        middle_pnt = [self.landmark[10].x, self.landmark[10].y] * dims

        ring_vec = [self.landmark[16].x - self.landmark[14].x,
                    self.landmark[16].y - self.landmark[14].y] * dims
        ring_pnt = [self.landmark[14].x, self.landmark[14].y] * dims

        pinky_vec = [self.landmark[20].x - self.landmark[18].x,
                     self.landmark[20].y - self.landmark[18].y] * dims
        pinky_pnt = [self.landmark[18].x, self.landmark[18].y] * dims

        draw_arrow(hand_vec, hand_pnt)
        draw_arrow(thumb_vec, thumb_pnt)
        draw_arrow(index_vec, index_pnt)
        draw_arrow(middle_vec, middle_pnt)
        draw_arrow(ring_vec, ring_pnt)
        draw_arrow(pinky_vec, pinky_pnt)


def dot(vec1, vec2):
    return np.vdot(vec1, vec2)


class HandRegion(Enum):
    PALM = 0,
    THUMB = 1,
    INDEX = 2,
    MIDDLE = 3,
    RING = 4,
    PINKY = 5
