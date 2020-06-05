import os
import sys

import yaml
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import distance
from copy import deepcopy

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
        self.num_detections = 0
        self.num_insertions = 0
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
        self.backgrounds = {}
        # self.mask_ids = list()
        self.masks_path = None
        # self.saved_masks = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right',
        #                                          'x_bot_left', 'y_bot_left', 'x_bot_right', 'y_bot_right'])
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

        print("FPS: ", self.fps)
        print(f"Time periods for processing: {self.config['periods']}")

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
        self.frame_num += 1

    def __detect_mask(self):
        rgb_frame = np.flip(self.frame, 2)
        result = self.model.detect([rgb_frame])[0]
        class_ids = result['class_ids']
        masks = result['masks']

        for i, class_id in enumerate(class_ids):
            if class_id in self.to_replace:
                mask = masks[:, :, i].astype(np.float32)

                mask = process_mask(mask)

                if mask.any():

                    banner_mask = np.zeros_like(rgb_frame)
                    points = np.where(mask == 1)
                    banner_mask[points] = rgb_frame[points]

                    contours = get_contours(banner_mask)

                    for cnt in contours:
                        if cv2.contourArea(cnt) > np.product(mask.shape) * 0.0008:

                            cv2.drawContours(self.frame, cnt, 3, (0, 255, 0), 3)
