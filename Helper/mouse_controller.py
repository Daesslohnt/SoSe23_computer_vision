from pynput.mouse import Button
from pynput import mouse

from HandRecognition.gesture import GestureClass

class MouseController:

    def __init__(self):
        self.mouse = mouse.Controller()
        self.rmb_pressed = False
        self.lmb_pressed = False
        self.mmb_pressed = False
        self.double_lmb_pressed = False
        self.rock_gesture_pressed = False

    def __del__(self):
        self.mouse.release(Button.left)
        self.mouse.release(Button.right)
        self.mouse.release(Button.middle)

    def move_cursor(self, movement):
        self.mouse.move(movement[0], movement[1])

    def scroll_down(self):
        self.mouse.scroll(0, -1)

    def scroll_up(self):
        self.mouse.scroll(0, 1)

    def mouse_button_left(self):
        if not self.lmb_pressed:
            self.mouse.press(Button.left)
            self.lmb_pressed = True
        if self.lmb_pressed:
            self.mouse.release(Button.left)
            self.lmb_pressed = False

    def mouse_button_right(self):
        if not self.rmb_pressed:
            self.mouse.press(Button.right)
            self.rmb_pressed = True
        if self.rmb_pressed:
            self.mouse.release(Button.right)
            self.rmb_pressed = False

    def mouse_button_middle(self):
        if not self.mmb_pressed:
            self.mouse.press(Button.middle)
            self.mmb_pressed = True
        if self.mmb_pressed:
            self.mouse.release(Button.middle)
            self.mmb_pressed = False

    def double_click(self):
        if not self.double_lmb_pressed:
            self.mouse.click(Button.left, 2)
            self.double_lmb_pressed = True
        if self.double_lmb_pressed:
            self.double_lmb_pressed = False

    def hold_on(self):
        if not self.rock_gesture_pressed:
            self.rock_gesture_pressed = True
        if self.rock_gesture_pressed:
            self.rock_gesture_pressed = False

    def gesture_action(self, gesture):
        if gesture == GestureClass.LEFT_CLICK:
            self.mouse_button_left()
        elif gesture == GestureClass.DOUBLE_CLICK:
            self.double_click()
        elif gesture == GestureClass.RIGHT_CLICK:
            self.mouse_button_right()
        elif gesture == GestureClass.SCROLL_UP:
            self.scroll_up()
        elif gesture == GestureClass.SCROLL_DOWN:
            self.scroll_down()
        elif gesture == GestureClass.SCROLL_WHEEL:
            self.mouse_button_middle()
        elif gesture == GestureClass.HOLD_ON:
            self.hold_on()