import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_points(df):

    def center(top_left, bot_right, bot_left, top_right):
        return ((top_left + bot_right) / 2 + (bot_left + top_right) / 2) / 2

    def euclidean_center(x_point, y_point):
        return np.sqrt((x_point-df['x_center'])**2 + (y_point-df['y_center'])**2)

    df.reset_index(drop=True, inplace=True)

    df['x_center'] = center(df['x_top_left'], df['x_bot_right'], df['x_bot_left'], df['x_top_right'])
    df['y_center'] = center(df['y_top_left'], df['y_bot_right'], df['y_bot_left'], df['y_top_right'])

    df['dist_top_left'] = euclidean_center(df['x_top_left'], df['y_top_left'])
    df['dist_bot_left'] = euclidean_center(df['x_bot_left'], df['y_bot_left'])
    df['dist_top_right'] = euclidean_center(df['x_top_right'], df['y_top_right'])
    df['dist_bot_right'] = euclidean_center(df['x_bot_right'], df['y_bot_right'])

    df['left_height'] = np.sqrt(
        (df['x_top_left'] - df['x_bot_left']) ** 2 + (
                    df['y_top_left'] - df['y_bot_left']) ** 2)
    df['top_width'] = np.sqrt(
        (df['x_top_left'] - df['x_top_right']) ** 2 + (
                    df['y_top_left'] - df['y_top_right']) ** 2)
    df['right_height'] = np.sqrt(
        (df['x_top_right'] - df['x_bot_right']) ** 2 + (
                    df['y_top_right'] - df['y_bot_right']) ** 2)
    df['bot_width'] = np.sqrt(
        (df['x_bot_left'] - df['x_bot_right']) ** 2 + (
                    df['y_bot_left'] - df['y_bot_right']) ** 2)

    df['ratio'] = df['top_width'] / (df['left_height'] + df['right_height'])/2
    ratio = np.median(df['ratio'])

    a = np.array(df[['x_top_right', 'y_top_right']])
    b = np.array(df[['x_bot_left', 'y_bot_left']])

    start_coordinate = np.array(df[['x_top_left', 'y_top_left']])

    a = a - start_coordinate
    b = b - start_coordinate

    df['cos_alpha'] = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    df['angle'] = np.arccos(df['cos_alpha']) * (180 / np.pi)

    lost_side = []
    prev_x = df.loc[0]
    for i in range(len(df)):
        data = df.loc[i]
        diff = data['dist_top_left'] - prev_x['dist_top_left']
        if abs(diff) > 6:
            lost_side.append(i)
        prev_x = data

    unstable_left = np.zeros(df.shape[0])
    unstable_right = np.zeros_like(unstable_left)

    for i in lost_side:
        if abs(df.loc[i, 'x_top_left'] - df.loc[i - 1, 'x_top_left']) > 10:
            unstable_left[i] = 1

        if abs(df.loc[i, 'x_top_right'] - df.loc[i - 1, 'x_top_right']) > 10:
            unstable_right[i] = 1

    df['unstable_right'] = unstable_right
    df['unstable_left'] = unstable_left

    x_top_left = df["x_top_left"]
    x_top_right = df["x_top_right"]
    y_top_right = df["y_top_right"]
    y_top_left = df["y_top_left"]


    def y_point(x_point):
        return (x_point - x_top_left) * (y_top_right - y_top_left) / (x_top_right - x_top_left) + y_top_left

    df['y_bot_right'] = y_point(df['x_bot_right']) + df["left_height"] * df["angle"] / 90
    df['y_bot_left'] = y_point(df['x_bot_left']) + df["left_height"] * df["angle"] / 90

    if df.shape[0] > 4:
        for x in range(df.shape[0]):
            row = df.loc[x]
            before = abs(x-df.index.min()) if abs(x-df.index.min()) < 11 else 11
            after = abs(df.index.max())-x if abs(df.index.max()-x) < 11 else 11
            if row['unstable_right']:
                for position in range(x - before, x + after):
                    x_top_left = df.loc[position, "x_top_left"]
                    x_bot_left = df.loc[position, "x_bot_left"]

                    tmp_x_top_right = x_top_left + df.loc[position, "left_height"] * (ratio * df.loc[position, "angle"] / 90)
                    tmp_x_bot_right = x_bot_left + df.loc[position, "left_height"] * (ratio * df.loc[position, "angle"] / 90)

                    df.loc[position, "x_top_right"] = tmp_x_top_right
                    df.loc[position, "x_bot_right"] = tmp_x_bot_right

            if row['unstable_left']:
                for position in range(x - before, x + after):
                    x_top_right = df.loc[position, "x_top_right"]
                    x_bot_right = df.loc[position, "x_bot_right"]

                    tmp_x_top_left = x_top_right - df.loc[position, "right_height"] * \
                                     (ratio * df.loc[position, "angle"] / 90)

                    tmp_x_bot_left = x_bot_right - df.loc[position, "right_height"] * \
                                     (ratio * df.loc[position, "angle"] / 90)

                    df.loc[position, "x_top_left"] = tmp_x_top_left
                    df.loc[position, "x_bot_left"] = tmp_x_bot_left

        df['y_top_left'] = smooth_series(df['y_top_left'])
        df['y_top_right'] = smooth_series(df['y_top_right'])
        df['y_bot_left'] = smooth_series(df['y_bot_left'])
        df['y_bot_right'] = smooth_series(df['y_bot_right'])

    df.drop(columns=['dist_top_right', 'dist_bot_right', 'dist_top_left', 'dist_bot_left',
                     'right_height', 'left_height', 'top_width', 'bot_width',
                     'unstable_left', 'unstable_right',
                     'cos_alpha', 'angle', 'ratio',
                     'x_center', 'y_center'], inplace=True)
    return df


def smooth_series(series):
    '''
    The method smoothes series of coordinates
    :series: series of coordinates to be smoothed
    :return: smoothed coordinates
    '''
    # load parameters
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
