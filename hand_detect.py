import asyncio

import cv2
import mediapipe as mp
import numpy as np

from Camera.Filter.blur_filter import BlurFilter
from Camera.Filter.flip_filter import FlipFilter
from Camera.webcam import Webcam
from HandRecognition.gesture import Gesture
from HandRecognition.hand import HandRegion, Hand
from Helper.mouse_controller import MouseController
from Helper.time_controller import TimeController
from model.freeze_graph import load_frozen_graph

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = load_frozen_graph(None)


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


async def do_gesture_action(hand, mouse_controller, gesture_sequence):
    predictions = model(hand.normalize_landmarks())[0].numpy()
    gesture_class = Gesture.define_gesture_class(predictions)
    gesture_sequence.append(gesture_class)
    if len(gesture_sequence) == 3:
        most_often = max(set(gesture_sequence), key=gesture_sequence.count)
        print(most_often)
        mouse_controller.gesture_action(most_often)
        gesture_sequence = []
    return gesture_sequence


async def main():
    # mouse = mouse.Controller()
    mouse_controller = MouseController()
    gesture_sequence = []

    old_pos = (0, 0)
    movement = (0, 0)
    scroll_timer = TimeController()
    movement_timer = TimeController()
    draw_rect = lambda rect: cv2.rectangle(image, rect.get_bounding_box()[0], rect.get_bounding_box()[1],
                                           (0, 0, 255), 2) if rect else None

    move_gesture = Gesture([lambda hand: hand.handedness == 'Right',
                            lambda hand: hand.is_extended(HandRegion.THUMB),
                            lambda hand: hand.is_extended(HandRegion.INDEX),
                            lambda hand: hand.is_extended(HandRegion.MIDDLE),
                            lambda hand: hand.is_extended(HandRegion.RING),
                            lambda hand: hand.is_extended(HandRegion.PINKY)
                            ])

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
                    task = asyncio.create_task(do_gesture_action(hand, mouse_controller, gesture_sequence))
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

                        mouse_controller.move_cursor(movement)

                    gesture_sequence = await task

                # Draw bones on image
                draw_bones(results, image)

            # Display image and check for exit (esc or 'q').
            cv2.imshow('MediaPipe Hands', image)

            key = cv2.waitKey(5)
            if key == 27 or key == 113:
                break
    del mouse_controller
    camera.release()


if __name__ == '__main__':
    asyncio.run(main())
