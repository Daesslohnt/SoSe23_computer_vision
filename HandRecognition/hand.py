from enum import Enum

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
        if finger is HandRegion.THUMB:
            return self.landmark[2].x > self.landmark[4].x
        elif finger is HandRegion.INDEX:
            return self.landmark[6].y > self.landmark[8].y
        elif finger is HandRegion.MIDDLE:
            return self.landmark[10].y > self.landmark[12].y
        elif finger is HandRegion.RING:
            return self.landmark[14].y > self.landmark[16].y
        elif finger is HandRegion.PINKY:
            return self.landmark[18].y > self.landmark[20].y
        else:
            return False


class HandRegion(Enum):
    PALM = 0,
    THUMB = 1,
    INDEX = 2,
    MIDDLE = 3,
    RING = 4,
    PINKY = 5
