import cv2
import numpy as np
from pynput.keyboard import Key, Controller
import math
import time

keyboard = Controller()
cam_number = 1

degree_threshold = 8
frame_threshold = 50
keypress_delta = 100
min_arrow_size = 1000
max_arrow_size = 8000
area_sum_offset = 0.6


def press_key(key, length):
    keyboard.press(key)
    time.sleep(length)
    keyboard.release(key)


def capture_video():
    capapture = cv2.VideoCapture(cam_number)
    if capapture.isOpened():
        ret, frame = capapture.read()
    else:
        ret = False

    name = "feed"
    cv2.namedWindow(name)

    frame_count = 0
    while ret:
        frame_count = frame_count + 1
        ret, frame = capapture.read()
        deg, image = detect_arrow(frame)

        deg = deg if abs(deg) > degree_threshold else 0
        next_turn = frame_threshold - abs(deg)
        if (next_turn < frame_threshold and (next_turn - frame_count) < 1):
            frame_count = 0
            key = Key.left if deg > 0 else Key.right
            press_key(key, abs(deg) / keypress_delta)

        cv2.imshow(name, image)
        if cv2.waitKey(1) == 27:
            break


def detect_arrow(image):
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, black_white_image = cv2.threshold(
        greyscale_image, 150, 255, cv2.THRESH_BINARY)
    black_white_image = cv2.medianBlur(black_white_image, 9)
    wireframe_image = cv2.Canny(black_white_image, 100, 200)
    contours, _ = cv2.findContours(
        wireframe_image,  cv2.RETR_TREE,  cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_arrow_size and area < max_arrow_size:
            approx = cv2.approxPolyDP(
                cnt, 0.03 * cv2.arcLength(cnt, True), True)
            if((len(approx) == 6)):
                for i in range(3):
                    arr1 = np.array([approx[i], approx[i + 1], approx[i + 2]])
                    arr2 = np.array(
                        [approx[i + 3], approx[(i + 4) % 6], approx[(i + 5) % 6]])
                    arr3 = np.array(approx)
                    area1 = cv2.contourArea(arr1)
                    area2 = cv2.contourArea(arr2)
                    area3 = cv2.contourArea(arr3)

                    if ((area1 + area2) > area_sum_offset*area3):
                        bigger = arr1 if area1 > area2 else arr2
                        myradians = math.atan2(
                            bigger[2][0][1] - bigger[0][0][1], bigger[2][0][0] - bigger[0][0][0])
                        deg = math.degrees(myradians) % 180
                        deg = deg if deg < 90 else deg - 180
                        cv2.drawContours(
                            image, [approx], 0, (0, 255, 0), thickness=6)
                        return deg, image
    return [0, image]


capture_video()
