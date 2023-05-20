import cv2
import mediapipe as mp
import numpy as np
from pynput import mouse
from pynput.mouse import Button

from Camera.Filter.blur_filter import BlurFilter
from Camera.Filter.flip_filter import FlipFilter
from Camera.webcam import Webcam
from HandRecognition.gesture import Gesture
from HandRecognition.hand import HandRegion, Hand, Direction
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
    double_lmb_pressed = False
    rock_gesture_pressed = False

    fast_gesture = Gesture([lambda hand: hand.handedness == 'Left',
                            lambda hand: hand.is_extended(HandRegion.INDEX),
                            lambda hand: hand.is_extended(HandRegion.MIDDLE)
                            ])

    move_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                            lambda hand: hand.is_extended(HandRegion.THUMB),
                            lambda hand: hand.is_extended(HandRegion.PINKY)
                            ])

    scroll_down_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                                   lambda hand: hand.points_towards(HandRegion.THUMB, Direction.DOWN),
                                   lambda hand: hand.is_extended(HandRegion.THUMB),
                                   lambda hand: not hand.is_extended(HandRegion.INDEX),
                                   lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                                   lambda hand: not hand.is_extended(HandRegion.RING),
                                   lambda hand: not hand.is_extended(HandRegion.PINKY)
                                   ])

    scrool_up_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                                 lambda hand: hand.points_towards(HandRegion.THUMB, Direction.UP),
                                 lambda hand: hand.is_extended(HandRegion.THUMB),
                                 lambda hand: not hand.is_extended(HandRegion.INDEX),
                                 lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                                 lambda hand: not hand.is_extended(HandRegion.RING),
                                 lambda hand: not hand.is_extended(HandRegion.PINKY)
                                 ])

    lmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: not hand.is_extended(HandRegion.INDEX),
                           lambda hand: hand.is_extended(HandRegion.MIDDLE),
                           lambda hand: hand.is_extended(HandRegion.PINKY)
                           ])

    rmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: hand.is_extended(HandRegion.INDEX),
                           lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                           lambda hand: hand.is_extended(HandRegion.PINKY)
                           ])

    mmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                           lambda hand: not hand.is_extended(HandRegion.RING),
                           lambda hand: hand.is_extended(HandRegion.PINKY)
                           ])

    double_lmb_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                                  lambda hand: not hand.is_extended(HandRegion.INDEX),
                                  lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                                  lambda hand: hand.is_extended(HandRegion.PINKY)
                                  ])

    rock_gesture = Gesture([lambda hand: hand.points_towards(HandRegion.INDEX, Direction.UP),
                            lambda hand: hand.is_extended(HandRegion.THUMB),
                            lambda hand: hand.is_extended(HandRegion.INDEX),
                            lambda hand: not hand.is_extended(HandRegion.MIDDLE),
                            lambda hand: not hand.is_extended(HandRegion.RING),
                            lambda hand: hand.is_extended(HandRegion.PINKY)]
                           )

    # For webcam input:
    camera = Webcam()
    camera.add_pipeline("flipped", [FlipFilter(), BlurFilter()])

    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
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
                    hand.draw_vecs(image, np.array([camera.width, camera.height]))

                    tp = 0.9

                    if move_gesture.is_gesture(hand):
                        # Calculate average position of hand
                        x = (hand.landmark[0].x + hand.landmark[5].x + hand.landmark[9].x + hand.landmark[13].x +
                             hand.landmark[17].x) / 5
                        y = (hand.landmark[0].y + hand.landmark[5].y + hand.landmark[9].y + hand.landmark[13].y +
                             hand.landmark[17].y) / 5
                        avg_position = [int(x * 640), int(y * 480)]

                        image = cv2.circle(img=image, center=avg_position, radius=20, color=(255, 0, 0), thickness=3)

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

                    # Scroll movement
                    if scroll_down_gesture.is_gesture(hand):
                        mouse.scroll(0, -1)

                    if scrool_up_gesture.is_gesture(hand):
                        mouse.scroll(0, 1)

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

                    # double klicks
                    if double_lmb_gesture.is_gesture(hand):
                        if not double_lmb_pressed:
                            mouse.click(Button.left, 2)
                            double_lmb_pressed = True
                    else:
                        if double_lmb_pressed:
                            double_lmb_pressed = False

                    # My Gesture
                    if rock_gesture.is_gesture(hand):
                        if not rock_gesture_pressed:
                            print("Lets Rock!")
                            rock_gesture_pressed = True
                    else:
                        if rock_gesture_pressed:
                            rock_gesture_pressed = False

                # Draw bones on image
                draw_bones(results, image)

            # Display image and check for exit (esc or 'q').
            cv2.imshow('MediaPipe Hands', image)

            key = cv2.waitKey(5)
            if key == 27 or key == 113:
                break

mouse.release(Button.left)
mouse.release(Button.right)
mouse.release(Button.middle)
camera.release()
