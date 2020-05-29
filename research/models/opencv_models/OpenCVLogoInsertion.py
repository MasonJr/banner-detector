import cv2 as cv
import numpy as np
from models.AbstractBannerReplacer import AbstractBannerReplacer
import yaml
import math
from models.opencv_models.banner_parameters_setting import banner_parameters_setting
from sklearn import metrics


class OpenCVLogoInsertion(AbstractBannerReplacer):
    """
    The model provides logo insertion with the OpenCV package
    """
    def __init__(self, template, frame, logo):
        self.template = template
        self.frame = frame
        self.logo = logo
        self.contours = []
        self.diagonal_coordinates_list = []
        self.corners = []
        self.template_p = {}
        self.f1_score = None

    def __detect_contour(self, matcher, min_match_count, dst_threshold, n_features, neighbours, rc_threshold):
        """
        The method provides detection of the field where the logo must be inserted

        :param matcher:  tunes the Matcher object
        :param min_match_count:  the minimum quantity of matched keypoints between frame and logo to detect the field
        :param dst_threshold: the threshold for the distance between matched descriptors
        :param n_features: the number of features for the SIFT algorithm
        :param neighbours: the amount of best matches found per each query descriptor
        :param rc_threshold: the threshold for the Homographies mask
        :return: switch that indicates whether the required field was found or not; the required field, the required
        field in hsv mode, cropped frame corners coordinates, cropped frame corner coordinates
        """
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.template = cv.imread(self.template)
        gray_template = cv.cvtColor(self.template, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(n_features)

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        index_params = {'algorithm': matcher['index_params'][0], 'trees': matcher['index_params'][1]}
        search_params = {'checks': matcher['search_params']}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=neighbours)

        good = []
        for m, n in matches:
            if m.distance < dst_threshold * n.distance:
                good.append(m)

        cr_frame = None
        frame_hsv = None
        min_max = None
        if len(good) >= min_match_count:
            switch = True
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, rc_threshold)
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, m)
            x_corner_list = [dst[i][0][0] for i in range(len(dst))]
            y_corner_list = [dst[j][0][1] for j in range(len(dst))]
            x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
            y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
            min_max = [x_min, x_max, y_min, y_max]
            cr_frame = self.frame[y_min:y_max, x_min:x_max]
            frame_hsv = cv.cvtColor(cr_frame, cv.COLOR_BGR2HSV)
        else:
            switch = False
        return switch, cr_frame, frame_hsv, min_max

    def __adjust_logo_color(self, required_field):
        """
        The method provides adjustment of the logo's color relative to the insertion field

        :param required_field: field for the logo insertion
        :return:
        """
        required_field = cv.cvtColor(required_field, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(required_field)
        mean_s = np.mean(s).astype(int)
        self.logo = cv.imread(self.logo)
        logo_hsv = cv.cvtColor(self.logo, cv.COLOR_BGR2HSV)
        logo_h, logo_s, logo_v = cv.split(logo_hsv)
        mean_logo_s = np.mean(logo_s).astype(int)
        s_coeff = round(mean_s / mean_logo_s, 2)
        new_s_logo = (logo_s * s_coeff).astype('uint8')
        new_logo_hsv = cv.merge([logo_h, new_s_logo, logo_v])
        self.logo = cv.cvtColor(new_logo_hsv, cv.COLOR_HSV2BGR)

    def __detect_banner_color(self, frame_hsv, h_params, s_params, v_params):
        """
        The method provides banners color detection and build contour of the detected figure

        :param frame_hsv: transformed frame to the HSV mode
        :param h_params: hue parameters for detecting the required color
        :param s_params: saturation parameters for detecting required color
        :param v_params: value parameters for detecting required color
        :return:
        """
        low_color = np.array([h_params['low'], s_params['low'], v_params['low']])
        high_color = np.array([h_params['high'], s_params['high'], v_params['high']])
        color_mask = cv.inRange(frame_hsv, low_color, high_color)
        _, self.contours, __ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def __find_diagonal_contour_coordinates(self, contour):
        """
        The method provides detection of the diagonal contour coordinates for further computations

        :param contour: the contour of the required field for logo insertion
        :return: the list of diagonal contour coordinates
        """
        x_top_left = contour[:, 0, 0].min()
        x_bot_right = contour[:, 0, 0].max()
        y_top_left = contour[:, 0, 1].min()
        y_bot_right = contour[:, 0, 1].max()
        self.diagonal_coordinates_list = [x_top_left, x_bot_right, y_top_left, y_bot_right]
        return self.diagonal_coordinates_list

    def __find_contour_coordinates(self, cr_frame, area_threshold, centroid_bias, y_coefficient):
        """
        The method provides detection of the contour corners coordinates

        :param cr_frame: field for the logo insertion
        :param area_threshold: the threshold for contour's area
        :param centroid_bias: deviation from contour centroid
        :param y_coefficient: coefficient for Y corners coordinates
        :return: banner mask, field for logo color adjustment
        """
        drop_list = [i for i in range(len(self.contours)) if cv.contourArea(self.contours[i]) < area_threshold]
        self.contours = [i for j, i in enumerate(self.contours) if j not in drop_list]

        area_centroids_list = []
        for cnt in self.contours:
            centroid = cv.moments(cnt)
            cx = int(centroid["m10"] / centroid["m00"])
            cy = int(centroid["m01"] / centroid["m00"])
            area_centroids_list.append([cv.contourArea(cnt), [cx, cy]])
        max_area = max([area[0] for area in area_centroids_list])
        y_centroid = [value[1][1] for value in area_centroids_list if value[0] == max_area]

        drop_list = [i for i, val in enumerate(area_centroids_list) if abs(val[1][1] - y_centroid[0]) > area_threshold]
        self.contours = [i for j, i in enumerate(self.contours) if j not in drop_list]

        for i in range(len(self.contours)):
            drop_list = [j for j in range(len(self.contours[i]))
                         if abs(self.contours[i][j][0][1] - y_centroid[0]) > centroid_bias]
            self.contours[i] = np.array([point for i, point in enumerate(self.contours[i]) if i not in drop_list])

        coordinates_array = []
        for cnt in self.contours:
            x_top_left, x_bot_right, y_top_left, y_bot_right = self.__find_diagonal_contour_coordinates(cnt)
            coordinates_array.append([x_top_left, x_bot_right, y_top_left, y_bot_right])

        x_top_left = min([coordinates_array[i][0] for i in range(len(coordinates_array))])
        x_bot_right = max([coordinates_array[i][1] for i in range(len(coordinates_array))])
        y_top_left = min([coordinates_array[i][2] for i in range(len(coordinates_array))])
        y_bot_right = max([coordinates_array[i][3] for i in range(len(coordinates_array))])

        banner_height = y_bot_right - y_top_left
        y_bot_left = y_bot_right - math.ceil(banner_height * y_coefficient)
        y_top_right = y_top_left + math.ceil(banner_height * y_coefficient)
        top_left = [x_top_left, y_top_left]
        bot_left = [x_top_left, y_bot_left]
        bot_right = [x_bot_right, y_bot_right]
        top_right = [x_bot_right, y_top_right]
        self.corners = [top_left, bot_left, bot_right, top_right]

        banner_mask_cr = cr_frame.copy()
        pts = np.array(self.corners, np.int32)
        cv.fillPoly(banner_mask_cr, [pts], (0, 0, 255), lineType=cv.LINE_AA)
        color_adj_field = cr_frame[y_top_left:y_bot_right, x_top_left:x_bot_right]
        return banner_mask_cr, color_adj_field

    def __adjust_referee_colors(self, hsv_referee, area_threshold, frame_hsv,
                                banner_mask_cr, coef, hsv_body, hsv_flag):
        """
        The method provides referee colors adjustment for flowing around the detected banner

        :param hsv_referee: h, s, v parameters for referee object
        :param area_threshold: the threshold for the contours area
        :param frame_hsv: transformed frame to the HSV mode
        :param banner_mask_cr: banner mask
        :param coef: coefficients for referee object tuning
        :param hsv_body: h, s, v parameters for referee's body
        :param hsv_flag: h, s, v parameters for flag
        :return:
        """
        low_ref = np.array([hsv_referee['low_h'], 0, hsv_referee['low_v']])
        high_ref = np.array([hsv_referee['high_h'], 255, hsv_referee['high_v']])
        referee_mask = cv.inRange(frame_hsv, low_ref, high_ref)

        _, contours, _ = cv.findContours(referee_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold[0]:
                cv.drawContours(banner_mask_cr, [cnt], 0, (0, 255, 0), -1)

                # body parts
                al = cnt[:, 0, 0].min()  # X coordinate for top left corner
                bl = cnt[:, 0, 0].max()  # X coordinate for bottom right corner
                cl = cnt[:, 0, 1].min()  # Y coordinate for top left corner
                dl = cnt[:, 0, 1].max()  # Y coordinate for bottom right corner

                ref_cr_hsv = frame_hsv[int(cl * coef['1']):int(dl * coef['2']), int(al * coef['3']):int(bl * coef['4'])]
                banner_mask_cr_ref = banner_mask_cr[int(cl * coef['1']):int(dl * coef['2']),
                                                    int(al * coef['3']):int(bl * coef['4'])]

                # body color
                low_bd = np.array([hsv_body['h'][0], hsv_body['s'][0], hsv_body['v'][0]])
                high_bd = np.array([hsv_body['h'][1], hsv_body['s'][1], hsv_body['v'][1]])
                body_mask = cv.inRange(ref_cr_hsv, low_bd, high_bd)

                _, contours, _ = cv.findContours(body_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                # drawing contour
                for cont in contours:
                    area = cv.contourArea(cont)
                    if area > area_threshold[1]:
                        cv.drawContours(banner_mask_cr_ref, [cont], 0, (0, 255, 0), -1)

        # flag color
        low_flg = np.array([hsv_flag['h'][0], hsv_flag['s'][0], hsv_flag['v'][0]])
        high_flg = np.array([hsv_flag['h'][1], hsv_flag['s'][1], hsv_flag['v'][1]])
        flag_mask = cv.inRange(frame_hsv, low_flg, high_flg)

        _, contours, _ = cv.findContours(flag_mask, cv.RETR_TREE,
                                         cv.CHAIN_APPROX_SIMPLE)
        # drawing contour
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1:
                cv.drawContours(banner_mask_cr, [cnt], 0, (0, 255, 0), 1)

    def __performance_evaluation(self, banner_mask_cr, f_name, min_max):
        """
        The method provide model performance evaluation

        :param banner_mask_cr: cropped frame
        :param f_name: frame name
        :param min_max: cropped frame corners coordinates
        :return:
        """
        f_name = f_name.split('.')

        try:
            frame_true = np.load('SET PATH/{}.npy'.format(f_name[0]))
            frame_true_cut = frame_true[min_max[2]:min_max[3], min_max[0]:min_max[1]]
        except FileNotFoundError:
            frame_true_cut = None

        if frame_true_cut is not None:
            cut_frame_mask = np.zeros_like(frame_true_cut)
            top_left, bot_left, bot_right, top_right = self.corners

            for i in range(top_left[1], bot_right[1]):
                for j in range(top_left[0], bot_right[0]):
                    if list(banner_mask_cr[i, j]) == [0, 0, 255]:
                        cut_frame_mask[i, j] = 1
            y_pred = cut_frame_mask.ravel()
            y_true = frame_true_cut.ravel()
            self.f1_score = metrics.f1_score(y_true, y_pred)
        else:
            self.f1_score = -1

    def __resize_banner(self, min_max, w_threshold, w_ratio):
        """
        The method provides banner resizing

        :param min_max: cropped frame corners coordinates
        :param w_threshold: frame width threshold
        :param w_ratio: height and width ratio
        :return: resized banner
        """
        top_left, bot_left, bot_right, top_right = self.corners

        w = bot_right[0] - top_left[0]
        h = bot_right[1] - top_left[1]

        if (bot_right[0] + min_max[0]) >= w_threshold * (self.frame.shape[1] - 1):
            w = math.ceil(h * w_ratio)

        resized_banner = cv.resize(self.logo, (w, h))
        rtx = [bot_right[0], top_left[1]]
        lbx = [top_left[0], bot_right[1]]
        pts1 = np.float32([[top_left, rtx, bot_right, lbx]])
        pts2 = np.float32([[top_left, top_right, bot_right, bot_left]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        resized_banner = cv.warpPerspective(resized_banner, matrix, (w, h), borderMode=1)
        resized_banner = cv.GaussianBlur(resized_banner, (1, 1), cv.BORDER_DEFAULT)
        return resized_banner

    def build_model(self, filename):
        """
        The method provides ability to set the required parameters for model building

        :param filename: file that contains required parameters for model tuning
        :return:
        """
        with open(filename, 'r') as stream:
            self.template_p = yaml.safe_load(stream)
        stream.close()

    def detect_banner(self, f_name):
        """
        The method provides detection of the required field and prepares it for replacement

        :param f_name: frame name
        :return: cropped field, resized banner, copy of cropped field, switch
        """
        parameter = self.template_p
        switch, cr_frame, frame_hsv, min_max = self.__detect_contour(parameter['matcher'],
                                                                     parameter['min_match_count'],
                                                                     parameter['dst_threshold'],
                                                                     parameter['n_features'],
                                                                     parameter['neighbours'],
                                                                     parameter['rc_threshold'])

        if switch:
            self.__detect_banner_color(frame_hsv, parameter['h_params'], parameter['s_params'],
                                       parameter['v_params'])

            banner_mask_cr, color_adjustment_field = self.__find_contour_coordinates(cr_frame,
                                                                                     parameter['cnt_area_threshold'],
                                                                                     parameter['centroid_bias'],
                                                                                     parameter['y_coefficient'])

            self.__adjust_logo_color(color_adjustment_field)

            self.__adjust_referee_colors(parameter['hsv_referee'], parameter['area_threshold'], frame_hsv,
                                         banner_mask_cr, parameter['coef'], parameter['hsv_body'],
                                         parameter['hsv_flag'])

            resized_banner = self.__resize_banner(min_max, parameter['w_threshold'], parameter['w_ratio'])

            if f_name is None:
                pass
            else:
                self.__performance_evaluation(banner_mask_cr, f_name, min_max)

            return cr_frame, resized_banner, banner_mask_cr, switch, self.f1_score
        else:
            return 0, 0, 0, switch, -1

    def insert_logo(self, cr_frame, resized_banner, banner_mask_cr, switch):
        """
        The method provides insertion of the required logo into the prepared field

        :param cr_frame: cropped field of frame
        :param resized_banner: resized banner
        :param banner_mask_cr: copy of cropped frame
        :param switch: indicates whether the required field was found or not
        :return:
        """
        if switch:
            top_left, bot_left, bot_right, top_right = self.corners

            for i in range(top_left[1], bot_right[1]):
                for j in range(top_left[0], bot_right[0]):
                    if list(banner_mask_cr[i, j]) == [0, 0, 255]:
                        cr_frame[i, j] = resized_banner[i - top_left[1], j - top_left[0]]
                    else:
                        continue
        else:
            pass
        cv.imshow('Replaced', self.frame)
