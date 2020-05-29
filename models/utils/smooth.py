import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter


def smooth_points(df):



    df.reset_index(drop=True, inplace=True)


    if df.shape[0] > 4:

        df['y_top_left'] = smooth_series(df['y_top_left'])
        df['y_top_right'] = smooth_series(df['y_top_right'])
        df['y_bot_left'] = smooth_series(df['y_bot_left'])
        df['y_bot_right'] = smooth_series(df['y_bot_right'])

        df['x_top_left'] = smooth_series(df['x_top_left'])
        df['x_top_right'] = smooth_series(df['x_top_right'])
        df['x_bot_left'] = smooth_series(df['x_bot_left'])
        df['x_bot_right'] = smooth_series(df['x_bot_right'])

    return df


def line_equation(left_point, right_point, new_point):
    x_left, y_left = left_point
    x_right, y_right = right_point
    return (new_point - x_left) * (y_right - y_left) / (x_right - x_left) + y_left


def process_mask(mask):
    center_right = None
    center_left = None

    mask_points = np.argwhere(mask == 1)
    if mask_points.any():

        max_x = mask_points[mask_points[:, 1].argmax()]
        min_x = mask_points[mask_points[:, 1].argmin()]
        top_y = mask_points[mask_points[:, 0].argmin()]

        if abs(max_x[0] - top_y[0]) > 10 and abs(min_x[0] - top_y[0]) > 10:

            left_y_shift, left_x_shift = np.abs(top_y - min_x)
            right_y_shift, right_x_shift = np.abs(top_y - max_x)

            left_angle = np.arctan(left_y_shift / left_x_shift) * 180 / np.pi
            right_angle = np.arctan(right_y_shift / right_x_shift) * 180 / np.pi

            if left_angle < right_angle:
                mask[:, top_y[1]:] = 0
            else:
                mask[:, :top_y[1]] = 0

        _, mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        first_cnt = True
        for msk_cnt in mask_contours:
            if cv2.contourArea(msk_cnt) > np.product(mask.shape) * 0.0012:
                rect = cv2.minAreaRect(msk_cnt)
                box = cv2.boxPoints(rect).astype(np.int)
                xm, ym = rect[0]
                if first_cnt:
                    first_cnt = False
                    left_ids = np.argwhere(box[:, 0] < xm).squeeze()
                    left = box[left_ids]
                    right = np.delete(box, np.s_[left_ids], 0)
                    top_left, bot_left = left[left[:, 1].argsort(axis=0)]
                    top_right, bot_right = right[right[:, 1].argsort(axis=0)]

                    center_left = xm
                    center_right = xm
                else:
                    left_ids = np.argwhere(box[:, 0] < xm).squeeze()
                    if xm < center_left:
                        left = box[left_ids]
                        top_left, bot_left = left[left[:, 1].argsort(axis=0)]
                        center_left = xm
                    elif xm > center_right:
                        right = np.delete(box, np.s_[left_ids], 0)
                        top_right, bot_right = right[right[:, 1].argsort(axis=0)]
                        center_right = xm

        if first_cnt:
            return []

        # right_height = bot_right[1] - top_right[1]
        # left_height = bot_left[1] - top_left[1]
        #
        # if top_right[0] > 1260 or bot_right[0] > 1260:
        #     start = np.min([top_right[0], bot_right[0]])
        #     restoring = np.arange(1200, 1280, 1).astype(np.int)
        #     top_y = line_equation(top_left, top_right, restoring).astype(np.int)
        #     for (x, y) in zip(restoring, top_y):
        #         mask[y:y + right_height, x] = 1
        # if top_left[0] < 20 or bot_left[0] < 20:
        #     finish = np.max([top_left[0], top_left[0]])
        #     restoring = np.arange(0, 80, 1).astype(np.int)
        #     top_y = line_equation(top_left, top_right, restoring).astype(np.int)
        #     for (x, y) in zip(restoring, top_y):
        #         mask[y:y + left_height, x] = 1

        mask_points = [*top_left, *top_right, *bot_left, *bot_right]

        return [mask, mask_points]


def smooth_series(series):

    min_window = 3
    max_window = 33
    poly_degree = 2
    threshold = 10

    best_series = []

    # smoothing
    for wnd_size in range(min_window, max_window):
        if wnd_size > len(series):
            break
        if wnd_size % 2 == 0:
            continue
        new_series = savgol_filter(series, wnd_size, poly_degree)
        if max(abs(new_series - series)) < threshold:
            best_series = new_series

    return best_series