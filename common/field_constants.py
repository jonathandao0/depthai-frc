# All units in meters
FIELD_HEIGHT = 8.2296
FIELD_WIDTH = 16.4592

ROBOT_SIZE = [0.5, 0.5]

LANDMARKS = {
    'red_upper_power_port_sandbox': {
        'name': 'red_upper_power_port_sandbox',
        'label_number': 5,
        # Image object width (in meters)
        'length': 0.8636,
        'width': 0.99695,
        # X, Y, Rot
        # Position is using image frame coordinates (Positive Y is down)
        'pose': [
            0.0,
            5.915393,
            2.49555,
            0
        ]
    },
    'red_lower_power_port_sandbox': {
        'name': 'red_lower_power_port_sandbox',
        'label_number': 4,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            0.0,
            5.915393,
            0.5842,
            0
        ]
    },
    'red_loading_bay_sandbox': {
        'name': 'red_loading_bay_sandbox',
        'label_number': 3,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            FIELD_WIDTH,
            5.736190,
            0.5842,         # TODO: UPDATE
            0
        ]
    },
    'red_station_1_sandbox': {
        'name': 'red_station_1_sandbox',
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
    'red_station_2_sandbox': {
        'name': 'red_station_2_sandbox',
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
    'red_station_3_sandbox': {
        'name': 'red_station_3_sandbox',
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
    'blue_upper_power_port_sandbox': {
        'name': 'blue_upper_power_port_sandbox',
        'label_number': 0,
        'length': 0.8636,
        'width': 0.99695,
        'pose': [
            FIELD_WIDTH,
            2.358665,
            2.49555,
            180
        ]
    },
    'blue_lower_power_port_sandbox': {
        'name': 'blue_lower_power_port_sandbox',
        'label_number': 1,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            FIELD_WIDTH,
            2.358665,
            0.5842,
            180
        ]
    },
    'blue_loading_bay_sandbox': {
        'name': 'blue_loading_bay_sandbox',
        'label_number': 6,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            0.0,
            2.358665,
            0.5842,         # TODO: UPDATE
            0
        ]
    },
    'blue_station_1_sandbox': {
        'name': 'blue_station_1_sandbox',
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
    'blue_station_2_sandbox': {
        'name': 'blue_station_2_sandbox',
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
    'blue_station_3_sandbox': {
        'name': 'blue_station_3_sandbox',
        'label_number': 99,
        'length': 0.99695,  # TODO: UPDATE
        'width': 2.7432,
        'pose': [
            0.390589,
            0.707132,
            0.5842,         # TODO: UPDATE
            20
        ]
    },
    'red_upper_power_port': {
        'name': 'red_upper_power_port',
        'label_number': 5,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,
        'pose': [
            0.0,
            5.915393,
            2.49555,
            0
        ]
    },
    'red_lower_power_port': {
        'name': 'red_lower_power_port',
        'label_number': 4,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            0.0,
            5.915393,
            0.5842,
            0
        ]
    },
    'red_loading_bay': {
        'name': 'red_loading_bay',
        'label_number': 3,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
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
        'pose': [
            FIELD_WIDTH,
            2.358665,
            2.49555,
            180
        ]
    },
    'blue_lower_power_port': {
        'name': 'blue_lower_power_port',
        'label_number': 1,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
        'pose': [
            FIELD_WIDTH,
            2.358665,
            0.5842,
            180
        ]
    },
    'blue_loading_bay': {
        'name': 'blue_loading_bay',
        'label_number': 6,
        'length': 0.99695,  # TODO: UPDATE
        'width': 0.99695,   # TODO: UPDATE
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


def pose3d_to_frame_position(object_pose, scaled_values):
    object_pose[1] = FIELD_HEIGHT - object_pose[1]
    pose = list(int(l * r) for l, r in zip(scaled_values, object_pose[:2]))
    # pose.append(object_pose[3])

    return pose


def object_pose_to_robot_pose(object_pose, camera_transform):
    robot_pose = object_pose
    return robot_pose


def robot_pose_to_frame_position(robot_pos, scaled_values):
    left   = robot_pos[0] - (ROBOT_SIZE[1] / 2)
    top    = FIELD_HEIGHT - robot_pos[1] - (ROBOT_SIZE[0] / 2)
    right  = robot_pos[0] + (ROBOT_SIZE[1] / 2)
    bottom = FIELD_HEIGHT - robot_pos[1] + (ROBOT_SIZE[0] / 2)

    left_top = pose3d_to_frame_position([left, top, 0, robot_pos[2]], scaled_values)
    right_bottom = pose3d_to_frame_position([right, bottom, 0, robot_pos[2]], scaled_values)

    return left_top, right_bottom


def to_wpilib_coords(cv2_pos):
    return cv2_pos[1], cv2_pos[0], cv2_pos[2]
