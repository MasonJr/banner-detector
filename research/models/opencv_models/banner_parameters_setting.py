import yaml


def banner_parameters_setting():
    function_list = []
    detect_contour_parameters = {'matcher': {'index_params': [1, 5], 'search_params': 70}, 'min_match_count': 10,
                                 'dst_threshold': 0.7, 'n_features': 200000, 'neighbours': 2, 'rc_threshold': 5.0}

    adjust_referee_colors_parameters = {'hsv_referee': {'low_h': 10, 'high_h': 180, 'low_v': 0, 'high_v': 100},
                                        'area_threshold': [100, 4], 'coef': {'1': 0.8, '2': 1.2, '3': 0.95, '4': 1.02},
                                        'hsv_body': {'h': [0, 200], 's': [10, 70], 'v': [150, 255]},
                                        'hsv_flag': {'h': [0, 50], 's': [50, 200], 'v': [150, 255]}}

    detect_banner_color_parameters = {'h_params': {'low': 105, 'high': 130}, 's_params': {'low': 40, 'high': 255},
                                      'v_params': {'low': 70, 'high': 255}}

    find_contour_coordinates_parameters = {'deviation': 0.1, 'cnt_area_threshold': 15, 'centroid_bias': 21,
                                           'y_coefficient': 0.242}

    resize_banner_parameters = {'w_threshold': 0.985, 'w_ratio': 5.7}

    function_list.extend([detect_contour_parameters, adjust_referee_colors_parameters,
                          detect_banner_color_parameters, find_contour_coordinates_parameters,
                          resize_banner_parameters])

    with open('../configurations/visa_parameters.yml', 'w') as outfile:
        for func in function_list:
            yaml.dump(func, outfile, default_flow_style=False)
        outfile.close()
