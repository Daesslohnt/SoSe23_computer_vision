import math
from enum import Enum

import cv2
import numpy
import numpy as np

from Helper.rectangle import Rectangle


class Vect:

    def __init__(self, tip, start):
        self.tip = tip
        self.start = start
        self.x = tip[0] - start[0]
        self.y = tip[1] - start[1]
        self.dir = float(np.degrees(np.arctan2(-self.x, -self.y)))
        self.len = self.veclen([self.x, self.y])
        self.x = self.x / self.len
        self.y = self.y / self.len

    def draw(self, image, dims=[640, 480], col=(255, 255, 255)):
        cv2.arrowedLine(image, [int(self.start[0] * dims[0]), int(self.start[1] * dims[1])], [int(self.tip[0] * dims[0]), int(self.tip[1] * dims[1])], col, thickness=2, tipLength=0.2)

    def veclen(self, vec):
        return np.sqrt(np.square(vec[0]) + np.square(vec[1]))

    def __str__(self):
        return "x=" + str(round(self.x, 3)) + "  y=" + str(round(self.y, 3)) + "  len=" + str(round(self.len, 3)) + "  dir=" + str(round(self.dir, 3)) + "°"


class Hand:

    def __init__(self, landmark, handedness):
        self.landmark = landmark
        self.handedness = handedness
        self.vec_palm_vertical = Vect([self.landmark[9].x, self.landmark[9].y], [self.landmark[0].x, self.landmark[0].y])
        self.vec_palm_horizontal = Vect([self.landmark[5].x, self.landmark[5].y], [self.landmark[17].x, self.landmark[17].y])
        self.vec_thumb_start = Vect([self.landmark[2].x, self.landmark[2].y], [self.landmark[1].x, self.landmark[1].y])
        self.vec_thumb_middle = Vect([self.landmark[3].x, self.landmark[3].y], [self.landmark[2].x, self.landmark[2].y])
        self.vec_thumb_tip = Vect([self.landmark[4].x, self.landmark[4].y], [self.landmark[3].x, self.landmark[3].y])
        self.vec_index_start = Vect([self.landmark[6].x, self.landmark[6].y], [self.landmark[5].x, self.landmark[5].y])
        self.vec_index_middle = Vect([self.landmark[7].x, self.landmark[7].y], [self.landmark[6].x, self.landmark[6].y])
        self.vec_index_tip = Vect([self.landmark[8].x, self.landmark[8].y], [self.landmark[7].x, self.landmark[7].y])
        self.vec_middle_start = Vect([self.landmark[10].x, self.landmark[10].y], [self.landmark[9].x, self.landmark[9].y])
        self.vec_middle_middle = Vect([self.landmark[11].x, self.landmark[11].y], [self.landmark[10].x, self.landmark[10].y])
        self.vec_middle_tip = Vect([self.landmark[12].x, self.landmark[12].y], [self.landmark[11].x, self.landmark[11].y])
        self.vec_ring_start = Vect([self.landmark[14].x, self.landmark[14].y], [self.landmark[13].x, self.landmark[13].y])
        self.vec_ring_middle = Vect([self.landmark[15].x, self.landmark[15].y], [self.landmark[14].x, self.landmark[14].y])
        self.vec_ring_tip = Vect([self.landmark[16].x, self.landmark[16].y], [self.landmark[15].x, self.landmark[15].y])
        self.vec_pinky_start = Vect([self.landmark[18].x, self.landmark[18].y], [self.landmark[17].x, self.landmark[17].y])
        self.vec_pinky_middle = Vect([self.landmark[19].x, self.landmark[19].y], [self.landmark[18].x, self.landmark[18].y])
        self.vec_pinky_tip = Vect([self.landmark[20].x, self.landmark[20].y], [self.landmark[19].x, self.landmark[19].y])

    def is_move(self):
        return self.is_flat() and self.is_thumb_straight() and self.is_index_straight() and self.is_middle_straight() and self.is_ring_straight()

    def is_left_button(self):
        return self.is_flat() and self.is_thumb_reverse() and self.is_index_straight() and self.is_middle_straight() and self.is_ring_straight()

    def is_double_click(self):
        return self.is_flat() and self.is_thumb_straight() and self.is_index_reverse() and self.is_middle_straight() and self.is_ring_straight()

    def is_right_button(self):
        return self.is_flat() and self.is_thumb_straight() and self.is_index_straight() and self.is_middle_reverse()

    def is_scroll_up(self):
        return self.is_back_up() and self.is_index_straight()

    def is_scroll_down(self):
        return self.is_back_down() and self.is_index_straight()

    def is_flat(self):
        return 45 > self.vec_palm_vertical.dir > -45 \
            and 100 > self.diffangle(self.vec_palm_horizontal.dir, self.vec_palm_vertical.dir) > 30 \
            and 1.0 < (self.vec_palm_vertical.len / self.vec_palm_horizontal.len) < 2.5 \
            and 0 > self.diffangle(self.vec_pinky_start.dir, self.vec_palm_vertical.dir) > -55 \
            and 0 > self.diffangle(self.vec_pinky_middle.dir, self.vec_palm_vertical.dir) > -55 \
            and 0 > self.diffangle(self.vec_pinky_tip.dir, self.vec_palm_vertical.dir) > -55

    def is_back_up(self):
        return 45 > self.vec_palm_vertical.dir > -45 \
            and 1.0 < (self.vec_palm_vertical.len / self.vec_palm_horizontal.len) < 2.5 \
            and -50 > self.diffangle(self.vec_palm_horizontal.dir, self.vec_palm_vertical.dir) > -100
    def is_back_down(self):
        return abs(self.vec_palm_vertical.dir) > 100 \
            and 1.0 < (self.vec_palm_vertical.len / self.vec_palm_horizontal.len) < 5.5 \
            and -50 > self.diffangle(self.vec_palm_horizontal.dir, self.vec_palm_vertical.dir) > -100
    def is_thumb_straight(self):
        return 90 > self.diffangle(self.vec_thumb_start.dir, self.vec_palm_vertical.dir) > 0 \
            and 90 > self.diffangle(self.vec_thumb_middle.dir, self.vec_palm_vertical.dir) > 0

    def is_thumb_reverse(self):
        return 30 > self.diffangle(self.vec_thumb_start.dir, self.vec_palm_vertical.dir) > -20 \
            and -10 > self.diffangle(self.vec_thumb_middle.dir, self.vec_palm_vertical.dir) > -80

    def is_index_straight(self):
        return 40 > self.diffangle(self.vec_index_start.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_index_middle.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_index_tip.dir, self.vec_palm_vertical.dir) > -40

    def is_index_reverse(self):
        angle = self.diffangle(self.vec_index_middle.dir, self.vec_palm_vertical.dir)
        return angle > 100 or angle < -100

    def is_middle_straight(self):
        return 40 > self.diffangle(self.vec_middle_start.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_middle_middle.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_middle_tip.dir, self.vec_palm_vertical.dir) > -40

    def is_middle_reverse(self):
        angle = self.diffangle(self.vec_middle_middle.dir, self.vec_palm_vertical.dir)
        return angle > 100 or angle < -100

    def is_ring_straight(self):
        return 40 > self.diffangle(self.vec_ring_start.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_ring_middle.dir, self.vec_palm_vertical.dir) > -40 \
            and 40 > self.diffangle(self.vec_ring_tip.dir, self.vec_palm_vertical.dir) > -40

    def is_ring_reverse(self):
        angle = self.diffangle(self.vec_ring_middle.dir, self.vec_palm_vertical.dir)
        return angle > 100 or angle < -100

    def is_pinky_straight(self):
        return 20 > self.diffangle(self.vec_pinky_start.dir, self.vec_palm_vertical.dir) > -55 \
            and 20 > self.diffangle(self.vec_pinky_middle.dir, self.vec_palm_vertical.dir) > -55 \
            and 20 > self.diffangle(self.vec_pinky_tip.dir, self.vec_palm_vertical.dir) > -55

    def is_pinky_reverse(self):
        angle = self.diffangle(self.vec_pinky_middle.dir, self.vec_palm_vertical.dir)
        return angle > 100 or angle < -100

    def test(self):
        pass
        # print("-------------------------------------------")
        # if self.is_move():
        #     print("move")
        # if self.is_left_button():
        #     print("left button")
        # if self.is_double_click():
        #     print("double click")
        # if self.is_right_button():
        #     print("right button")
        # if self.is_scroll_up():
        #     print("scroll up")
        # if self.is_scroll_down():
        #     print("scroll down")
        # print("is flat         =" + str(self.is_flat()))
        # print("is back up      =" + str(self.is_back_up()))
        # print("is back down    =" + str(self.is_back_down()))
        # print("thumb straight  =" + str(self.is_thumb_straight()))
        # print("thumb reverse   =" + str(self.is_thumb_reverse()))
        # print("index straight  =" + str(self.is_index_straight()))
        # print("index reverse   =" + str(self.is_index_reverse()))
        # print("middle straight =" + str(self.is_middle_straight()))
        # print("middle reverse  =" + str(self.is_middle_reverse()))
        # print("ring straight   =" + str(self.is_ring_straight()))
        # print("ring reverse    =" + str(self.is_ring_reverse()))
        # print("pinky straight  =" + str(self.is_pinky_straight()))
        # print("pinky reverse   =" + str(self.is_pinky_reverse()))
        # print("-------------------------------------------")

    def diffangle(self, angle1, angle2):
        angle = angle1 - angle2
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360
        return angle

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

    def is_twisted(self):
        palm_vec_vertical = [self.landmark[9].x - self.landmark[0].x,
                             self.landmark[9].y - self.landmark[0].y] * np.array([640, 480])
        palm_vec_horizontal = [self.landmark[5].x - self.landmark[17].x,
                               self.landmark[5].y - self.landmark[17].y] * np.array([640, 480])
        # print(length(palm_vec_vertical) / length(palm_vec_horizontal))
        return not 1 < length(palm_vec_vertical) / length(palm_vec_horizontal) < 3

    def is_facing_camera(self):
        return self.landmark[5].x < self.landmark[17].x

    def is_extended(self, finger):
        palm_vec = [self.landmark[9].x - self.landmark[0].x,
                    self.landmark[9].y - self.landmark[0].y]
        palm_vec_horizontal = [self.landmark[5].x - self.landmark[13].x,
                               self.landmark[5].y - self.landmark[13].y]

        if finger is HandRegion.THUMB:
            thumb_vec = [self.landmark[3].x - self.landmark[2].x,
                         self.landmark[3].y - self.landmark[2].y]
            return dot(palm_vec_horizontal, thumb_vec) > 0

            # intersection_pnt = get_intersect([self.landmark[3].x, self.qlandmark[3].y],
            #                                  [self.landmark[2].x, self.landmark[2].y],
            #                                  [self.landmark[0].x, self.landmark[0].y],
            #                                  [self.landmark[9].x, self.landmark[9].y])
            # helper_vec = [intersection_pnt[0] - self.landmark[0].x,
            #               intersection_pnt[1] - self.landmark[0].y]
            # return dot(palm_vec, helper_vec) < 0
        elif finger is HandRegion.INDEX:
            index_vec = [self.landmark[8].x - self.landmark[6].x,
                         self.landmark[8].y - self.landmark[6].y]
            return dot(palm_vec, index_vec) > 0
        elif finger is HandRegion.MIDDLE:
            middle_vec = [self.landmark[12].x - self.landmark[10].x,
                          self.landmark[12].y - self.landmark[10].y]
            return dot(palm_vec, middle_vec) > 0
        elif finger is HandRegion.RING:
            ring_vec = [self.landmark[16].x - self.landmark[14].x,
                        self.landmark[16].y - self.landmark[14].y]
            return dot(palm_vec, ring_vec) > 0
        elif finger is HandRegion.PINKY:
            pinky_vec = [self.landmark[20].x - self.landmark[18].x,
                         self.landmark[20].y - self.landmark[18].y]
            return dot(palm_vec, pinky_vec) > 0
        else:
            return False

    def draw_vecs(self, image, dims):
        self.vec_thumb_tip.draw(image, dims=dims, col=(255, 255, 255))
        self.vec_index_tip.draw(image, dims=dims, col=(255, 255, 255))
        self.vec_middle_tip.draw(image, dims=dims, col=(255, 255, 255))
        self.vec_ring_tip.draw(image, dims=dims, col=(255, 255, 255))
        self.vec_pinky_tip.draw(image, dims=dims, col=(255, 255, 255))
        self.vec_thumb_middle.draw(image, dims=dims, col=(0, 255, 0))
        self.vec_index_middle.draw(image, dims=dims, col=(0, 255, 0))
        self.vec_middle_middle.draw(image, dims=dims, col=(0, 255, 0))
        self.vec_ring_middle.draw(image, dims=dims, col=(0, 255, 0))
        self.vec_pinky_middle.draw(image, dims=dims, col=(0, 255, 0))
        self.vec_thumb_start.draw(image, dims=dims, col=(79, 165, 255))
        self.vec_index_start.draw(image, dims=dims, col=(79, 165, 255))
        self.vec_middle_start.draw(image, dims=dims, col=(79, 165, 255))
        self.vec_ring_start.draw(image, dims=dims, col=(79, 165, 255))
        self.vec_pinky_start.draw(image, dims=dims, col=(79, 165, 255))
        self.vec_palm_horizontal.draw(image, dims=dims, col=(0, 255, 255))
        self.vec_palm_vertical.draw(image, dims=dims, col=(0, 255, 255))

    def points_towards(self, hand_region, direction):
        region_vec = self._get_region_vec(hand_region, True)
        if math.fabs(region_vec[0]) > math.fabs(region_vec[1]):
            if region_vec[0] > 0:
                return Direction.RIGHT == direction
            else:
                return Direction.LEFT == direction
        else:
            if region_vec[1] < 0:
                return Direction.UP == direction
            else:
                return Direction.DOWN == direction

    def _get_region_vec(self, hand_region, tip=False):
        if not tip:
            if hand_region == HandRegion.PALM:
                return [self.landmark[9].x - self.landmark[0].x,
                        self.landmark[9].y - self.landmark[0].y]
            elif hand_region == HandRegion.THUMB:
                return [self.landmark[4].x - self.landmark[2].x,
                        self.landmark[4].y - self.landmark[2].y]
            elif hand_region == HandRegion.INDEX:
                return [self.landmark[8].x - self.landmark[6].x,
                        self.landmark[8].y - self.landmark[6].y]
            elif hand_region == HandRegion.MIDDLE:
                return [self.landmark[12].x - self.landmark[10].x,
                        self.landmark[12].y - self.landmark[10].y]
            elif hand_region == HandRegion.RING:
                return [self.landmark[16].x - self.landmark[14].x,
                        self.landmark[16].y - self.landmark[14].y]
            elif hand_region == HandRegion.PINKY:
                return [self.landmark[20].x - self.landmark[18].x,
                        self.landmark[20].y - self.landmark[18].y]
        else:
            if hand_region == HandRegion.PALM:
                return [self.landmark[9].x - self.landmark[0].x,
                        self.landmark[9].y - self.landmark[0].y]
            elif hand_region == HandRegion.THUMB:
                return [self.landmark[4].x - self.landmark[3].x,
                        self.landmark[4].y - self.landmark[3].y]
            elif hand_region == HandRegion.INDEX:
                return [self.landmark[8].x - self.landmark[7].x,
                        self.landmark[8].y - self.landmark[7].y]
            elif hand_region == HandRegion.MIDDLE:
                return [self.landmark[12].x - self.landmark[11].x,
                        self.landmark[12].y - self.landmark[11].y]
            elif hand_region == HandRegion.RING:
                return [self.landmark[16].x - self.landmark[15].x,
                        self.landmark[16].y - self.landmark[15].y]
            elif hand_region == HandRegion.PINKY:
                return [self.landmark[20].x - self.landmark[19].x,
                        self.landmark[20].y - self.landmark[19].y]


def dot(vec1, vec2):
    return np.vdot(vec1, vec2)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float('inf'), float('inf'))
    return (x / z, y / z)


def length(vec):
    return math.sqrt(sum([x * x for x in vec]))


class HandRegion(Enum):
    PALM = 0,
    THUMB = 1,
    INDEX = 2,
    MIDDLE = 3,
    RING = 4,
    PINKY = 5


class Direction(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
