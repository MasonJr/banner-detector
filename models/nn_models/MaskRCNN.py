import os
import sys

import yaml
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import distance

sys.path.append('../models/mrcnn')
from models.nn_models.mrcnn.config import Config
from models.utils.mask_processing import found_corners, get_contours, create_background
from collections import defaultdict
from models.utils.smooth import smooth_points, process_mask


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # number of classes (we would normally add +1 for the background)
    NUM_CLASSES = 1 + 6
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    # network
    BACKBONE = "resnet50"
    # Learning rate
    LEARNING_RATE = 0.006
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # setting Max ground truth instances
    MAX_GT_INSTANCES = 14


class MRCNNLogoInsertion:

    def __init__(self):
        self.model = None
        self.frame = None
        self.frame_num = 0
        self.load_smooth = True
        self.load_smooth_mask = True
        self.detection_successful = False
        self.corners = None
        self.replace = None
        self.fps = None
        self.key = None
        self.start = None
        self.finish = None
        self.config = None
        self.process = False
        self.to_replace = None
        self.point_ids = list()
        self.class_match = defaultdict(list)
        self.before_smoothing = True
        self.mask_id = None
        self.class_ids = list()
        self.mask_ids = list()
        self.masks_path = None
        self.saved_masks = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right',
                                                 'x_bot_left', 'y_bot_left', 'x_bot_right', 'y_bot_right'])
        self.cascade_mask = defaultdict(dict)
        self.saved_points = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right',
                                                  'x_bot_left', 'y_bot_left', 'x_bot_right', 'y_bot_right'])

    def init_params(self, params):

        with open(params) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.replace = self.config['replace']
        self.to_replace = list(self.replace.keys())
        self.masks_path = self.config['mask_path']

        if bool(self.config['periods']):
            self.key = list(self.config['periods'].keys())[0]
            self.start = int(self.config['periods'][self.key]['start'])
            self.finish = int(self.config['periods'][self.key]['finish'])
        else:
            self.process = True

    def __valid_time(self):

        if self.key:
            times = self.frame_num / self.fps
            if (self.start <= times) and (times <= self.finish):
                self.process = True
            else:
                self.process = False

            if times == self.finish:
                print(f"Ended {self.key.split('_')[0]} {self.key.split('_')[1]}")
                del self.config['periods'][self.key]
                if len(self.config['periods'].keys()):
                    self.key = list(self.config['periods'].keys())[0]
                    self.start = int(self.config['periods'][self.key]['start'])
                    self.finish = int(self.config['periods'][self.key]['finish'])

    def detect_banner(self, frame):

        self.frame = frame
        self.__valid_time()
        if self.process:
            if self.before_smoothing:
                self.__detect_mask()
            else:
                if self.frame_num in self.class_match:
                    self.detection_successful = True
                else:
                    self.detection_successful = False
        self.frame_num += 1

    def __detect_mask(self):
        rgb_frame = np.flip(self.frame, 2)
        result = self.model.detect([rgb_frame])[0]
        class_ids = result['class_ids']
        masks = result['masks']

        mask_id = 0

        for i, class_id in enumerate(class_ids):
            if class_id in self.to_replace:
                mask = masks[:, :, i].astype(np.float32)

                mask_output = process_mask(mask)

                if mask_output:
                    mask, mask_points = mask_output

                    self.mask_ids.append((self.frame_num, i))
                    self.saved_masks.loc[f"{self.frame_num}_{i}"] = mask_points

                    banner_mask = np.zeros_like(rgb_frame)
                    points = np.where(mask == 1)
                    banner_mask[points] = rgb_frame[points]

                    contours = get_contours(banner_mask)

                    tmp_mask_id = []
                    for cnt in contours:
                        if cv2.contourArea(cnt) > np.product(mask.shape) * 0.0008:
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect).astype(np.int)

                            cnt_corners = found_corners(box)

                            self.point_ids.append((self.frame_num, mask_id))
                            self.saved_points.loc[f"{self.frame_num}_{mask_id}"] = cnt_corners
                            tmp_mask_id.append(mask_id)
                            mask_id += 1

                    self.class_match[self.frame_num].append({i: class_id})
                    self.cascade_mask[self.frame_num][i] = tmp_mask_id

                    np.save(os.path.join(self.masks_path, f'frame_{self.frame_num}_{i}.npy'), mask.astype(np.int8))

    def __get_smoothed_points(self, is_mask=False):

        def center(top_left, bot_right, bot_left, top_right):
            return (top_left + bot_right + bot_left + top_right) / 4

        if is_mask:
            mask_ind = pd.MultiIndex.from_tuples(self.mask_ids, names=('frame_num', 'original_mask_id'))
            self.saved_masks.index = mask_ind
            saved_corners = self.saved_masks.copy(deep=True)
            center_thresh = 50
        else:
            mind = pd.MultiIndex.from_tuples(self.point_ids, names=('frame_num', 'mask_id'))
            self.saved_points.index = mind
            saved_corners = self.saved_points.copy(deep=True)
            center_thresh = 30

        smooth_df = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right',
                                          'x_bot_left', 'y_bot_left', 'x_bot_right', 'y_bot_right'])

        while saved_corners.shape[0]:
            smooth_idx = []

            prev_frame_num = saved_corners.index[0]
            prev_points = saved_corners.loc[prev_frame_num]
            prev_center_x = center(prev_points[0], prev_points[6], prev_points[4], prev_points[2])
            prev_center_y = center(prev_points[1], prev_points[7], prev_points[5], prev_points[3])

            saved_corners.drop(prev_frame_num, inplace=True)

            smooth_df.loc[prev_frame_num[0]] = list(prev_points)
            smooth_idx.append(prev_frame_num)

            for frame_num, points in saved_corners.iterrows():
                if frame_num[0] - prev_frame_num[0] == 1:
                    center_x = center(points[0], points[6], points[4], points[2])
                    center_y = center(points[1], points[7], points[5], points[3])
                    dist = distance.euclidean([prev_center_x, prev_center_y], [center_x, center_y])
                    if dist < center_thresh:
                        smooth_df.loc[frame_num[0]] = list(points)
                        smooth_idx.append(frame_num)
                        saved_corners.drop(frame_num, inplace=True)

                        prev_center_x = center_x
                        prev_center_y = center_y
                        prev_frame_num, prev_points = frame_num, points

                elif frame_num[0] - prev_frame_num[0] > 1:
                    break

            smooth_df = smooth_df.astype(np.float32)
            # smooth_df = smooth_points(smooth_df)
            if is_mask:
                # smooth_df = smooth_points(smooth_df)
                smooth_idx = pd.MultiIndex.from_tuples(smooth_idx, names=('frame_num', 'original_mask_id'))
                smooth_df.index = smooth_idx
                self.saved_masks.loc[smooth_idx] = smooth_df
            else:
                smooth_df = smooth_points(smooth_df)
                smooth_idx = pd.MultiIndex.from_tuples(smooth_idx, names=('frame_num', 'mask_id'))
                smooth_df.index = smooth_idx
                self.saved_points.loc[smooth_idx] = smooth_df

            smooth_df.drop(smooth_idx, inplace=True)

    def __load_points(self):
        '''
        The method loads smoothed points
        '''
        if self.load_smooth:
            self.__get_smoothed_points()
            self.load_smooth = False

        row = np.array(self.saved_points.loc[(self.frame_num - 1, self.mask_id)])

        self.corners = np.split(row, 4)

    def __load_mask(self, original_mask_id):

        if self.load_smooth_mask:
            self.__get_smoothed_points(is_mask=True)
            self.load_smooth_mask = False

        row = np.array(self.saved_masks.loc[(self.frame_num - 1, original_mask_id)])
        mask_path = os.path.join(self.masks_path, f'frame_{self.frame_num - 1}_{original_mask_id}.npy')
        mask = np.load(mask_path).astype(np.uint8)
        mask[:, :int(row[0])] = 0
        mask[:, int(row[2]):] = 0

        os.remove(mask_path)

        return mask

    def insert_logo(self):
        '''
        This method insert logo into detected area on the frame
        '''
        # load logo
        if not self.detection_successful or not self.process:
            return

        frame_num = self.frame_num - 1
        matching = self.class_match[frame_num]
        cascades = self.cascade_mask[frame_num]
        frame_h, frame_w = self.frame.shape[:2]

        backgrounds = dict()
        banners = np.unique([list(class_match.values())[0] for class_match in matching])

        for class_id in banners:
            backgrounds[class_id] = create_background(self.replace[class_id], self.frame.shape)

        for match in matching:
            main_mask_id, class_id = match.popitem()
            mask = self.__load_mask(main_mask_id)
            submasks = cascades[main_mask_id]

            for mask_id in submasks:
                self.mask_id = mask_id
                logo = cv2.imread(self.replace[class_id], cv2.IMREAD_UNCHANGED)

                if logo.shape[2] == 4:
                    logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)

                self.__load_points()
                transformed_logo, box = self.__adjust_logo_shape(logo)
                box = box.reshape(4, 2).astype(np.int32)
                zero = np.zeros((frame_h, frame_w))
                zero = cv2.drawContours(zero, [box], -1, (1), -1)
                points = np.where(zero == 1)
                backgrounds[class_id][points] = transformed_logo[points]

            mask_points = np.where(mask == 1)
            self.frame[mask_points] = backgrounds[class_id][mask_points]

        del self.class_match[frame_num]
        del self.cascade_mask[frame_num]

    def __adjust_logo_shape(self, logo):

        # points before and after transformation
        # top_left, bot_left, bot_right, top_right
        h, w = logo.shape[:2]
        pts1 = np.float32([(0, 0), (0, (h - 1)), ((w - 1), (h - 1)), ((w - 1), 0)])
        pts2 = np.float32([self.corners[0], self.corners[2], self.corners[3], self.corners[1]])

        # perspective transformation
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_logo = cv2.warpPerspective(logo, mtrx, (self.frame.shape[1], self.frame.shape[0]), borderMode=1)

        return transformed_logo, pts2

    def __logo_color_adj(self, logo):

        # select banner area
        banner = self.frame[int(self.corners[0][1]):int(self.corners[2][1]),
                 int(self.corners[0][0]):int(self.corners[1][0])].copy()

        # get logo hsv
        logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = np.transpose(logo_hsv, (2, 0, 1))

        # get banner hsv
        banner_hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
        _, banner_s, _ = np.transpose(banner_hsv, (2, 0, 1))

        # find the saturation difference between both images
        mean_logo_s = np.mean(logo_s).astype(int)
        mean_banner_s = np.mean(banner_s).astype(int)
        trans_coef = round(mean_banner_s / mean_logo_s, 2)

        # adjust logo saturation according to the difference
        adjusted_logo_s = (logo_s * trans_coef).astype('uint8')
        adjusted_logo_hsv = np.array([logo_h, adjusted_logo_s, logo_v]).transpose((1, 2, 0))
        adjusted_logo = cv2.cvtColor(adjusted_logo_hsv, cv2.COLOR_HSV2BGR)

        return adjusted_logo
