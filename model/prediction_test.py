import keras as k
import numpy as np
import mediapipe as mp
import cv2
import os

from Camera.Filter.blur_filter import BlurFilter
from Camera.Filter.flip_filter import FlipFilter
from Camera.webcam import Webcam

path = os.path.join("..", "daten", "default", "WIN_20230527_18_28_41_Pro.jpg")
mp_hands = mp.solutions.hands
model = k.models.load_model("best_model.h5")

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
                if classi.classification[0].label == "Right":
                    landmarks = results.multi_hand_landmarks[index].landmark
                    landmarks = np.array([[lm.x, lm.y] for lm in landmarks])


                    # normalization
                    item = landmarks.T
                    x, y = item
                    x_max, x_min = max(x), min(x)
                    y_max, y_min = max(y), min(y)
                    x = (x - x_min) / (x_max - x_min)
                    y = (y - y_min) / (y_max - y_min)
                    item[0] = x
                    item[1] = y

                    landmarks = item.T
                    landmarks = np.reshape(landmarks, (1, 21, 2))

                    prediction = model.predict(landmarks, verbose=0)

                    index = np.argmax(prediction[0])
                    if prediction[0][index] < 0.8 and index != 0:
                        index = 7

                    print('-'*30)
                    if index == 0:
                        print("left_click")
                    elif index == 1:
                        print("double click")
                    elif index == 2:
                        print("right click")
                    elif index == 3:
                        print("scroll up")
                    elif index == 4:
                        print("scroll down")
                    elif index == 5:
                        print("scroll wheel")
                    elif index == 6:
                        print("hold on")
                    else:
                        print("default")