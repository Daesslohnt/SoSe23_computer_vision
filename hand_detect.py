import cv2
import mediapipe as mp
from pynput import mouse

from rectangle import Rectangle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if __name__ == "__main__":

    mouse = mouse.Controller()
    pfinal = [0, 0]

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

                        draw(palm)
                        draw(thumb)
                        draw(index)
                        draw(middle)
                        draw(ring)
                        draw(pinkie)

                        # Calculate average position of wrist,thumb and pinkie
                        x = (landmark[0].x + landmark[4].x + landmark[20].x) / 3
                        y = (landmark[0].y + landmark[4].y + landmark[20].y) / 3
                        circle_center = [int(x * 640), int(y * 480)]

                        image = cv2.circle(img=image, center=circle_center, radius=20, color=(255, 0, 0), thickness=3)

                        # depth pass
                        tp = 0.3
                        pfinal[0] = int(pfinal[0] * (1 - tp) + circle_center[0] * tp)
                        pfinal[1] = int(pfinal[1] * (1 - tp) + circle_center[1] * tp)
                        mouse.position = [int((pfinal[0] - 20) * 3.5), int((pfinal[1] - 20) * 3)]

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
