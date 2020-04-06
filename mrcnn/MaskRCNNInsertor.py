import json

import yaml
import numpy as np
import cv2

from config import Config
import model as modellib
from collections import defaultdict


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # number of classes (we would normally add +1 for the background)
    NUM_CLASSES = 1+8
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    # network
    BACKBONE = "resnet50"
    # Learning rate
    LEARNING_RATE=0.006
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10


class MRCNNLogoInsertion():

    def __init__(self):
        self.model = None
        self.frame = None
        self.masks = None
        self.frame_num = 0
        self.detection_successful = False
        self.frame = None
        self.corners = None
        self.replace = None
        self.center_left = None
        self.center_right = None
        self.fps = None
        self.key = None
        self.start = None
        self.finish = None
        self.period = None
        self.config = None
        self.process = False
        self.frame_corners = defaultdict(dict)
        self.masks = defaultdict(dict)
        # self.saved_points = []
        self.frame_masks = defaultdict(dict)

    def init_params(self, params):
        """
        reading parameters in dictionary
        :param params:
        :return:
        """

        with open(params) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.replace = self.config['replace']

        self.key = list(self.config['periods'].keys())[0]
        self.period = self.config['periods'][self.key]
        self.start, self.finish = self.period.values()

    def detect_banner(self, frame):

        '''
        This method detects banner's pixels using Unet model, and saves deteсted binary mask
        and saves coordinates for top left and bottom right corners of a banner
        :frame: image or video frame where we will make detection and insertion
        '''

        self.frame = frame
        self.__valid_time()
        self.masks.clear()
        self.frame_corners.clear()
        if self.process:
            self.__detect_mask()
            for class_id in self.masks:
                for mask_id in self.masks[class_id]:
                    mask = self.masks[class_id][mask_id]
                    self.__check_contours(mask, class_id, mask_id)

            # print(self.frame_masks)
            # self.saved_points.append(self.frame_masks)

        self.frame_num += 1

    def __valid_time(self):
        """
        checks time intervals
        :return:
        """
        time = self.frame_num / self.fps
        if self.start <= time and time <= self.finish:
            print("Detecting")
            self.process = True
        else:
            self.process = False

        if time == self.finish:
            del self.config['periods'][self.key]
            if len(self.config['periods'].keys()):
                self.key = list(self.config['periods'].keys())[0]
                self.period = self.config['periods'][self.key]
                self.start, self.finish = self.period.values()

    def __detect_mask(self):
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.model.detect([rgb_frame])
        r = results[0]
        for i, class_id in enumerate(r['class_ids']):
            if class_id in self.replace.keys():
                mask = r['masks'][:, :, i].astype(np.float32)
                self.masks[class_id][i] = mask

    def __check_contours(self, fsz_mask, class_id, mask_id):
        '''
        This method finding detected contours and corner coordinates
        :fsz_mask: detected full size mask
        '''
        # load parameters
        filter_area_size = self.config['filter_area_size']

        # finding contours
        first_cnt = True
        _, thresh = cv2.threshold(fsz_mask, 0.5, 255, 0)
        thresh = thresh.astype(np.uint8)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > filter_area_size:

                # looking for coorner points
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.float32(box)

                # X and Y coordinates for center of detected rectangle
                xm, ym = rect[0]

                # detecting coordinates for each corner
                # works for the first contour
                if first_cnt:
                    first_cnt = False
                    for point in box:
                        if point[0] < xm:
                            if point[1] < ym:
                                top_left = point
                            else:
                                bot_left = point
                        else:
                            if point[1] < ym:
                                top_right = point
                            else:
                                bot_right = point

                    self.center_left = xm
                    self.center_right = xm

                # works with more than one contour, and replace coordinates with more relevant
                else:
                    # left side
                    if xm < self.center_left:
                        for point in box:
                            if point[0] < xm:
                                if point[1] < ym:
                                    top_left = point
                                else:
                                    bot_left = point
                        self.center_left = xm

                        # right side
                    elif xm > self.center_right:
                        for point in box:
                            if point[0] > xm:
                                if point[1] < ym:
                                    top_right = point
                                else:
                                    bot_right = point
                        self.center_right = xm

                # fill spaces in contours
                cv2.drawContours(fsz_mask, [cnt], -1, (1), -1)

        # return if there is no detected area
        if first_cnt:
            np.save('saved_frame_mask/frame_{}.npy'.format(self.frame_num, ), np.zeros(1, dtype=np.uint8))
            return

        # saving detected mask
        np.save('saved_frame_mask/frame_{}_{}.npy'.format(self.frame_num, mask_id), fsz_mask)

        # saving corner points to dataframe
        self.frame_masks[self.frame_num][mask_id] = [int(i) for i in [top_left[0], top_left[1], top_right[0],
                                                     top_right[1], bot_left[0], bot_left[1],
                                                     bot_right[0], bot_right[1]]]

        self.frame_corners[class_id][mask_id] = self.frame_masks[self.frame_num][mask_id]




    def __load_points(self, class_id, mask_id):
        '''
        The method loads smoothed points
        '''
        # getiing points
        top_left = (self.frame_corners[class_id][mask_id][0], self.frame_corners[class_id][mask_id][1])
        top_right = (self.frame_corners[class_id][mask_id][2], self.frame_corners[class_id][mask_id][3])
        bot_left = (self.frame_corners[class_id][mask_id][4], self.frame_corners[class_id][mask_id][5])
        bot_right = (self.frame_corners[class_id][mask_id][6], self.frame_corners[class_id][mask_id][7])

        # saving coordinates
        self.corners = [top_left, bot_right, top_right, bot_left]


    def insert_logo(self):
        '''
        This method insert logo into detected area on the frame
        '''
        # load logo
        for banner_id in self.replace:
            logo = cv2.imread(self.replace[banner_id], cv2.IMREAD_UNCHANGED)
            if banner_id in self.frame_corners:
                for mask_id in self.frame_corners[banner_id]:
                    self.__load_points(banner_id, mask_id)
                    logo = self.__logo_color_adj(logo)
                    transformed_logo = self.__adjust_logo_shape(logo)

                    for k in range(self.frame.shape[0]):
                        for j in range(self.frame.shape[1]):
                            if self.masks[banner_id][mask_id][k, j] == 1:
                                self.frame[k, j] = transformed_logo[k, j]

    def __adjust_logo_shape(self, logo):

        '''
        The method resizes and applies perspective transformation on logo
        :logo: the logo that we will transform
        :return: transformed logo
        '''

        # points before and after transformation
        # top_left, bot_left, bot_right, top_right
        pts1 = np.float32(
            [(0, 0), (0, (logo.shape[0] - 1)), ((logo.shape[1] - 1), (logo.shape[0] - 1)),
             ((logo.shape[1] - 1), 0)])
        pts2 = np.float32([self.corners[0], self.corners[3], self.corners[1], self.corners[2]])

        # perspective transformation
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_logo = cv2.warpPerspective(logo, mtrx, (self.frame.shape[1], self.frame.shape[0]), borderMode=1)

        return transformed_logo

    def __logo_color_adj(self, logo):

        '''
        The method changes color of the logo to adjust it to frame
        :logo: the logo that we will change
        :return: changed logo
        '''

        # select banner area
        banner = self.frame[int(self.corners[0][1]):int(self.corners[1][1]),
                 int(self.corners[0][0]):int(self.corners[1][0])].copy()

        # get logo hsv
        logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv2.split(logo_hsv)

        # get banner hsv
        banner_hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
        _, banner_s, _ = cv2.split(banner_hsv)

        # find the saturation difference between both images
        mean_logo_s = np.mean(logo_s).astype(int)
        mean_banner_s = np.mean(banner_s).astype(int)
        trans_coef = round(mean_banner_s / mean_logo_s, 2)

        # adjust logo saturation according to the difference
        adjusted_logo_s = (logo_s * trans_coef).astype('uint8')
        adjusted_logo_hsv = cv2.merge([logo_h, adjusted_logo_s, logo_v])
        adjusted_logo = cv2.cvtColor(adjusted_logo_hsv, cv2.COLOR_HSV2BGR)

        return adjusted_logo


if __name__ == '__main__':

    logo_insertor = MRCNNLogoInsertion()
    logo_insertor.init_params("template.yaml")

    config = myMaskRCNNConfig()
    logo_insertor.model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    logo_insertor.model.load_weights(logo_insertor.config['model_weight_path'], by_name=True)
    # load parameters
    source_type = logo_insertor.config['source_type']
    source_link = logo_insertor.config['source_link']
    save_result = logo_insertor.config['save_result']
    saving_link = logo_insertor.config['saving_link']

    if source_type == 0:
        cap = cv2.VideoCapture(source_link)
        logo_insertor.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(saving_link, four_cc, logo_insertor.fps, (frame_width, frame_height), True)
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                logo_insertor.detect_banner(frame)
                logo_insertor.insert_logo()
                out.write(frame)
                # cv2.imshow('Video (press Q to close)', frame)
                if cv2.waitKey(23) & 0xFF == ord('q'):
                    break
            else:
                break
        saved_points = json.dumps(dict(logo_insertor.frame_masks))
        with open('saved_points.txt', 'w') as f:
            f.write(saved_points)
        cap.release()
        cv2.destroyAllWindows()
        out.release()

    else:
        frame = cv2.imread(source_link, cv2.IMREAD_UNCHANGED)

        logo_insertor.detect_banner(frame)

        logo_insertor.frame_num = 0
        logo_insertor.before_smoothing = False

        logo_insertor.detect_banner(frame)
        logo_insertor.insert_logo()

        if save_result:
            cv2.imwrite(saving_link, frame)
        cv2.imshow('Image (press Q to close)', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
