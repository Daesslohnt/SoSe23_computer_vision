import time

import cv2
import mediapipe as mp
from pynput import mouse
from pynput.mouse import Button

from rectangle import Rectangle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if __name__ == "__main__":

    mouse = mouse.Controller()
    pfinal = [0, 0]
    old_pos = (0, 0)
    movement = (0, 0)
    last_perf_time_mouse = 0
    last_perf_time_scroll = 0

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:
        while cap.isOpened():
            success, image = cap.read()

            image = cv2.flip(image, 1)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:

                # Identify right hand, read positions, calculate depth pass and set cursor position
                for index, classi in enumerate(results.multi_handedness):
                    if classi.classification[0].label == 'Right':
                        draw = lambda rect: cv2.rectangle(image, rect.get_bounding_box()[0],
                                                          rect.get_bounding_box()[1], (0, 0, 255), 2)
                        landmark = results.multi_hand_landmarks[index].landmark

                        # Create and then draw bounding boxes for the parts of the hand
                        palm = Rectangle([(landmark[i].x, landmark[i].y) for i in [0, 5, 9, 13, 17]], [640, 480])
                        thumb = Rectangle([(landmark[i].x, landmark[i].y) for i in range(2, 5)], [640, 480])
                        index = Rectangle([(landmark[i].x, landmark[i].y) for i in range(6, 9)], [640, 480])
                        middle = Rectangle([(landmark[i].x, landmark[i].y) for i in range(10, 13)], [640, 480])
                        ring = Rectangle([(landmark[i].x, landmark[i].y) for i in range(14, 17)], [640, 480])
                        pinkie = Rectangle([(landmark[i].x, landmark[i].y) for i in range(18, 21)], [640, 480])

                        thumb_extended = not palm.collides(thumb)
                        index_extended = not palm.collides_y(index)
                        middle_extended = not palm.collides_y(middle)
                        ring_extended = not palm.collides_y(ring)
                        pinkie_extended = not palm.collides_y(pinkie)

                        draw(palm)
                        draw(thumb)
                        draw(index)
                        draw(middle)
                        draw(ring)
                        draw(pinkie)

                        #alternative way
                        thumb_extended = landmark[2].x > landmark[4].x
                        index_extended = landmark[6].y > landmark[8].y
                        middle_extended = landmark[10].y > landmark[12].y
                        ring_extended = landmark[14].y > landmark[16].y
                        pinkie_extended = landmark[18].y > landmark[20].y

                        # Calculate average position of wrist,thumb and pinkie
                        x = (landmark[0].x + landmark[4].x + landmark[20].x) / 3
                        y = (landmark[0].y + landmark[4].y + landmark[20].y) / 3
                        circle_center = [int(x * 640), int(y * 480)]

                        image = cv2.circle(img=image, center=circle_center, radius=20, color=(255, 0, 0), thickness=3)

                        tp = 0.9
                        if pinkie_extended:
                            if thumb_extended:
                                # mouse movement

                                # time since last movement
                                perf_time = time.perf_counter()
                                mouse_frametime = perf_time - last_perf_time_mouse
                                print("move:",mouse_frametime)
                                last_perf_time_mouse = perf_time

                                # move mouse
                                x *= 640
                                y *= 480

                                if mouse_frametime > 0.1:
                                    old_pos = (x, y)

                                fact = 2
                                movement = (movement[0] * (1 - tp) + (x - old_pos[0]) * tp) * fact, \
                                    (movement[1] * (1 - tp) + (y - old_pos[1]) * tp) * fact
                                old_pos = (x, y)

                                mouse.move(movement[0], movement[1])
                            else:
                                # mouse scrolling

                                # time since last movement
                                perf_time = time.perf_counter()
                                scroll_frametime = perf_time - last_perf_time_scroll
                                print("scro:" ,scroll_frametime)
                                last_perf_time_scroll = perf_time


                                # scroll
                                x *= 640
                                y *= 480

                                if scroll_frametime > 0.1:
                                    old_pos = (x, y)

                                fact = 2
                                movement = (movement[0] * (1 - tp) + (x - old_pos[0]) * tp) * fact, \
                                    (movement[1] * (1 - tp) + (y - old_pos[1]) * tp) * fact
                                old_pos = (x, y)

                                scroll_up = movement[1] > 0

                                mouse.scroll(0,- movement[1])




                            # mouse button left
                            if not index_extended:
                                mouse.press(Button.left)
                            else:
                                mouse.release(Button.left)

                            #mouse button right
                            if not middle_extended:
                                mouse.press(Button.right)
                            else:
                                mouse.release(Button.right)

                        # mouse button middle
                        if not ring_extended:
                            mouse.press(Button.middle)
                        else:
                            mouse.release(Button.middle)

                        # depth pass
                        pfinal[0] = int(pfinal[0] * (1 - tp) + circle_center[0] * tp)
                        pfinal[1] = int(pfinal[1] * (1 - tp) + circle_center[1] * tp)
                        # mouse.position = [int((pfinal[0] - 20) * 3.5), int((pfinal[1] - 20) * 3)]

                # print landmarks to screen
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
