from enum import Enum
from numpy import argmax

class Gesture:

    def __init__(self, conditions: [lambda hand: bool]):
        self.conditions = conditions

    def is_gesture(self, hand) -> bool:
        for condition in self.conditions:
            if not condition(hand):
                return False
        return True

    @staticmethod
    def define_gesture_class(predictions):
        most_possible = argmax(predictions[0])
        if predictions[0][most_possible] < 0.9 and most_possible != 0:
            most_possible = 7
        if most_possible == 0 and predictions[0][most_possible] < 0.6:
            most_possible = 7
        if most_possible == 0:
            return GestureClass.LEFT_CLICK
        elif most_possible == 1:
            return GestureClass.DOUBLE_CLICK
        elif most_possible == 2:
            return GestureClass.RIGHT_CLICK
        elif most_possible == 3:
            return GestureClass.SCROLL_UP
        elif most_possible == 4:
            return GestureClass.SCROLL_DOWN
        elif most_possible == 5:
            return GestureClass.SCROLL_WHEEL
        elif most_possible == 6:
            return GestureClass.HOLD_ON
        else:
            return GestureClass.DEFAULT

class GestureClass(Enum):
    LEFT_CLICK = 0,
    DOUBLE_CLICK = 1,
    RIGHT_CLICK = 2,
    SCROLL_UP = 3,
    SCROLL_DOWN = 4,
    SCROLL_WHEEL = 5,
    HOLD_ON = 6,
    DEFAULT = 7
