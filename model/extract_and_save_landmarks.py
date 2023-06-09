import mediapipe as mp
import numpy as np
import os
import cv2

"""
    For each gesture it's own directory:
    left click
    double click
    right click
    scroll up
    scroll down
    scroll wheel click
    hold on
    default
"""

mp_hands = mp.solutions.hands

IMG_DIR = os.path.join("..", "daten")
LEFT_CLICK_DIR = os.path.join(IMG_DIR, "left_click")
DOUBLE_CLICK_DIR = os.path.join(IMG_DIR, "double_click")
RIGHT_CLICK_DIR = os.path.join(IMG_DIR, "right_click")
SCROLL_UP_DIR = os.path.join(IMG_DIR, "scroll_up")
SCROLL_DOWN_DIR = os.path.join(IMG_DIR, "scroll_down")
SCROLL_WHEEL_CLICK_DIR = os.path.join(IMG_DIR, "scroll_wheel_click")
HOLD_ON_DIR = os.path.join(IMG_DIR, "hold_on")
DEFAULT_DIR = os.path.join(IMG_DIR, "default")

left_click_files = [os.path.join(LEFT_CLICK_DIR, name) for name in os.listdir(LEFT_CLICK_DIR)]
double_click_files = [os.path.join(DOUBLE_CLICK_DIR, name) for name in os.listdir(DOUBLE_CLICK_DIR)]
right_click_files = [os.path.join(RIGHT_CLICK_DIR, name) for name in os.listdir(RIGHT_CLICK_DIR)]
scroll_up_files = [os.path.join(SCROLL_UP_DIR, name) for name in os.listdir(SCROLL_UP_DIR)]
scroll_down_files = [os.path.join(SCROLL_DOWN_DIR, name) for name in os.listdir(SCROLL_DOWN_DIR)]
scroll_wheel_click_files = [os.path.join(SCROLL_WHEEL_CLICK_DIR, name) for name in os.listdir(SCROLL_WHEEL_CLICK_DIR)]
hold_on_files = [os.path.join(HOLD_ON_DIR, name) for name in os.listdir(HOLD_ON_DIR)]
default_files = [os.path.join(DEFAULT_DIR, name) for name in os.listdir(DEFAULT_DIR)]

all_files = {
    "left_click": left_click_files,
    "double_click": double_click_files,
    "right_click": right_click_files,
    "scroll_up": scroll_up_files,
    "scroll_down": scroll_down_files,
    "scroll_wheel_click": scroll_wheel_click_files,
    "hold_on": hold_on_files,
    "default": default_files
}

X = []
y = []

with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        max_num_hands=2) as hands:

    for label, files in all_files.items():


        for file in files:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for index, class_i in enumerate(results.multi_handedness):
                    hand_label = class_i.classification[0].label
                    hand_landmarks = results.multi_hand_landmarks[index].landmark

                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.append([landmark.x, landmark.y])

                    X.append(landmarks)

                    label_code = [0]*8
                    if label == "left_click":
                        label_code[0] = 1
                    elif label == "double_click":
                        label_code[1] = 1
                    elif label == "right_click":
                        label_code[2] = 1
                    elif label == "scroll_up":
                        label_code[3] = 1
                    elif label == "scroll_down":
                        label_code[4] = 1
                    elif label == "scroll_wheel_click":
                        label_code[5] = 1
                    elif label == "hold_on":
                        label_code[6] = 1
                    else:
                        label_code[7] = 1
                    assert sum(label_code) == 1, "wrong classification"
                    y.append(label_code)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(IMG_DIR, "X-H.npy"), X)
np.save(os.path.join(IMG_DIR, "y-H.npy"), y)