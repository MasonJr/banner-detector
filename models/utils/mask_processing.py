import numpy as np
import cv2


def found_corners(box):

    box.view('i8,i8').sort(order=['f0'], axis=0)

    left_side = box[:2]
    right_side = box[2:]

    left_bot_y = np.argmax(left_side, axis=0)[1]
    left_top_y = np.argmin(left_side, axis=0)[1]
    right_bot_y = np.argmax(right_side, axis=0)[1]
    right_top_y = np.argmin(right_side, axis=0)[1]

    top_left = left_side[left_top_y]
    bot_left = left_side[left_bot_y]
    top_right = right_side[right_top_y]
    bot_right = right_side[right_bot_y]

    return [*top_left, *top_right, *bot_left, *bot_right]


def get_contours(mask):

    small_kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    gray_erosion = cv2.erode(gray_mask, small_kernel, iterations=1)
    gray_dilation = cv2.dilate(gray_erosion, small_kernel, iterations=5)
    ret, thresh = cv2.threshold(gray_dilation, 127, 256, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)
    gauss = cv2.GaussianBlur(opening, (5, 5), 1)
    dilation = cv2.dilate(gauss, small_kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=3)

    _, contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def create_background(logo_path, frame_shape):
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    if logo.shape[2] == 4:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)

    un, cnts = np.unique(logo, axis=1, return_counts=True)
    idx = cnts.argmax()
    pixel_value = logo[:, idx][0]
    background = np.full(frame_shape, pixel_value)

    return background