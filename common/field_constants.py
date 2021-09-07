# All units in meters
FIELD_HEIGHT = 8.2296
FIELD_WIDTH = 16.4592

ROBOT_SIZE = [0.5, 0.5]

LANDMARKS = {
    # 'red_upper_power_port_sandbox': {
    #     'name': 'red_upper_power_port_sandbox',
    #     'label_number': 5,
    #     # Image object width (in meters)
    #     'length': 0.8636,
    #     'width': 0.99695,
    #     'good_matches_threshold': 7,
    #     # X, Y, Z Rot
    #     # Position is 2D field image frame coordinates (Positive Y is down)
    #     'pose': [
    #         0.0,
    #         5.915393,
    #         2.49555,
    #         0
    #     ],
    #     # 3D coordinate of the top-left corner of the ref image. Use this to generate 3D points from ref image keypoints
    #     'ref_image_pose': [
    #         0.0,
    #         5.915393,
    #         2.49555,
    #         0
    #     ],
    #     # # Image (0,0) point in world 3D coords. Used to generate keypoint coords
    #     # 'image_corner_pose': [
    #     #     0.0,
    #     #     5.915393,
    #     #     2.49555,
    #     # ],
    #     # 'ppm':
    # },
    'red_upper_power_port': {
        'name': 'red_upper_power_port',
        'label_number': 5,
        'length': 0.8636,
        'width': 0.99695,
        'good_matches_threshold': 7,
        'pose': [
            0.0,
            5.915393,
            2.49555,
            180
        ]
    },
    'red_lower_power_port': {
        'name': 'red_lower_power_port',
        'label_number': 4,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'good_matches_threshold': 3,
        'pose': [
            0.0,
            5.915393,
            0.5842,
            180
        ]
    },
    'red_loading_bay': {
        'name': 'red_loading_bay',
        'label_number': 3,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'good_matches_threshold': 10,
        'pose': [
            FIELD_WIDTH,
            5.736190,
            0.5842,         # TODO: UPDATE
            0
        ]
    },
    'red_station_1': {
        'name': 'red_station_1',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            16.082352,
            0.707132,
            0.5842,         # TODO: UPDATE
            160
        ]
    },
    'red_station_2': {
        'name': 'red_station_2',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            FIELD_WIDTH,
            3.812507,
            0.5842,         # TODO: UPDATE
            180
        ]
    },
    'red_station_3': {
        'name': 'red_station_3',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            16.041130,
            7.426284,
            0.5842,         # TODO: UPDATE
            200
        ]
    },
    'blue_upper_power_port': {
        'name': 'blue_upper_power_port',
        'label_number': 0,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,
        'good_matches_threshold': 7,
        'pose': [
            FIELD_WIDTH,
            2.358665,
            2.49555,
            0
        ]
    },
    'blue_lower_power_port': {
        'name': 'blue_lower_power_port',
        'label_number': 1,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'good_matches_threshold': 3,
        'pose': [
            FIELD_WIDTH,
            2.358665,
            0.5842,
            0
        ]
    },
    'blue_loading_bay': {
        'name': 'blue_loading_bay',
        'label_number': 6,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'good_matches_threshold': 10,
        'pose': [
            0.0,
            2.358665,
            0.5842,         # TODO: UPDATE
            0
        ]
    },
    'blue_station_1': {
        'name': 'blue_station_1',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.363108,
        'pose': [
            16.082352,
            7.426284,
            0.5842,         # TODO: UPDATE
            340
        ]
    },
    'blue_station_2': {
        'name': 'blue_station_2',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            0.0,
            4.183503,
            0.5842,         # TODO: UPDATE
            0
        ]
    },
    'blue_station_3': {
        'name': 'blue_station_3',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            0.390589,
            0.707132,
            0.5842,         # TODO: UPDATE
            20
        ]
    }
}

