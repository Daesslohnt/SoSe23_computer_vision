import math

import cv2
import mediapipe as mp
from pynput import mouse
from pynput.mouse import Button

from Camera.Filter.flip_filter import FlipFilter
from Camera.webcam import Webcam
from HandRecognition.gesture import Gesture
from HandRecognition.hand import HandRegion, Hand
from Helper.time_controller import TimeController

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def draw_bones(results, image):
    # print landmarks to screen
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


if __name__ == "__main__":
    mouse = mouse.Controller()

    old_pos = (0, 0)
    movement = (0, 0)
    scroll_timer = TimeController()
    movement_timer = TimeController()
    draw_rect = lambda rect: cv2.rectangle(image, rect.get_bounding_box()[0], rect.get_bounding_box()[1],
                                           (0, 0, 255), 2) if rect else None
    rmb_pressed = False
    lmb_pressed = False
    mmb_pressed = False

    move_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                            lambda hand: hand.is_extended(HandRegion.THUMB),
                            lambda hand: hand.is_extended(HandRegion.PINKY)
                            ])

    scroll_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                              lambda hand: not hand.is_extended(HandRegion.THUMB),
                              lambda hand: hand.is_extended(HandRegion.PINKY)
                              ])

    lmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: not hand.is_extended(HandRegion.INDEX),
                           lambda hand: hand.is_extended(HandRegion.PINKY)
                           ])

    rmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                           lambda hand: hand.is_extended(HandRegion.PINKY)
                           ])

    mmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: not hand.is_extended(HandRegion.RING),
                           lambda hand: hand.is_extended(HandRegion.PINKY)])

    # For webcam input:
    camera = Webcam()
    camera.add_pipeline("flipped", [FlipFilter()])
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:
        while True:
            image = camera.get_image("flipped")

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:

                # Gesture recognition and mouse input
                for index, classi in enumerate(results.multi_handedness):
                    hand = Hand(results.multi_hand_landmarks[index].landmark, classi.classification[0].label)

                    if hand.handedness == 'Right':
                        draw_rect(hand.get_bounding_box(HandRegion.PALM))

                        # Calculate average position of wrist,thumb and pinkie
                        x = (hand.landmark[0].x + hand.landmark[4].x + hand.landmark[20].x) / 3
                        y = (hand.landmark[0].y + hand.landmark[4].y + hand.landmark[20].y) / 3
                        avg_position = [int(x * 640), int(y * 480)]

                        image = cv2.circle(img=image, center=avg_position, radius=20, color=(255, 0, 0), thickness=3)

                        tp = 0.9

                        if move_gesture.is_gesture(hand):
                            # mouse movement

                            x *= 640
                            y *= 480

                            if movement_timer.is_new_activation():
                                old_pos = (x, y)

                            fact = 2
                            movement = (movement[0] * (1 - tp) + (x - old_pos[0]) * tp) * fact, \
                                       (movement[1] * (1 - tp) + (y - old_pos[1]) * tp) * fact
                            old_pos = (x, y)

                            mouse.move(movement[0], movement[1])

                        if scroll_gesture.is_gesture(hand):
                            # mouse scrolling

                            x *= 640
                            y *= 480

                            if scroll_timer.is_new_activation():
                                old_pos = (x, y)

                            fact = 2
                            movement = (movement[0] * (1 - tp) + (x - old_pos[0]) * tp) * fact, \
                                       (movement[1] * (1 - tp) + (y - old_pos[1]) * tp) * fact
                            old_pos = (x, y)

                            sign = math.copysign(1, movement[1])
                            scroll_movement = sign * (math.floor(math.fabs(movement[1])) / 10)
                            print(scroll_movement)
                            mouse.scroll(0, scroll_movement)

                        # mouse button left
                        if lmb_gesture.is_gesture(hand):
                            if not lmb_pressed:
                                mouse.press(Button.left)
                                lmb_pressed = True
                        else:
                            if lmb_pressed:
                                mouse.release(Button.left)
                                lmb_pressed = False

                        # mouse button right
                        if rmb_gesture.is_gesture(hand):
                            if not rmb_pressed:
                                mouse.press(Button.right)
                                rmb_pressed = True
                        else:
                            if rmb_pressed:
                                mouse.release(Button.right)
                                rmb_pressed = False

                        # mouse button middle
                        if mmb_gesture.is_gesture(hand):
                            if not mmb_pressed:
                                mouse.press(Button.middle)
                                mmb_pressed = True
                        else:
                            if mmb_pressed:
                                mouse.release(Button.middle)
                                mmb_pressed = False

                # Draw bones on image
                draw_bones(results, image)

            # Display image and check for exit.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

mouse.release(Button.left)
mouse.release(Button.right)
mouse.release(Button.middle)
camera.release()
