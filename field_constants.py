# All units in meters
FIELD_HEIGHT = 8.2296
FIELD_WIDTH = 16.4592

# X, Y, Rot
RED_POWER_PORT = (0.0, FIELD_HEIGHT - 5.915393, 0)
BLUE_POWER_PORT = (FIELD_WIDTH, FIELD_HEIGHT - 2.358665, 180)

RED_LOADING_BAY = (FIELD_WIDTH, FIELD_HEIGHT - 5.736190, 180)
BLUE_LOADING_BAY = (0.0, FIELD_HEIGHT - 2.520890, 0)

RED_STATION_1 = (16.082352, FIELD_HEIGHT - 0.707132, 160)
RED_STATION_2 = (FIELD_WIDTH, FIELD_HEIGHT - 3.812507, 180)
RED_STATION_3 = (16.041130, FIELD_HEIGHT - 7.426284, 200)

BLUE_STATION_1 = (0.363108, FIELD_HEIGHT - 7.426284, 340)
BLUE_STATION_2 = (0.0, FIELD_HEIGHT - 4.183503, 0)
BLUE_STATION_3 = (0.390589, FIELD_HEIGHT - 0.707132, 20)

ROBOT_SIZE = [0.5, 0.5]


def robot_position_to_frame_coords(robot_pos):
    top    = robot_pos[0] - (ROBOT_SIZE[0] / 2)
    left   = robot_pos[1] - (ROBOT_SIZE[1] / 2)
    bottom = robot_pos[0] + (ROBOT_SIZE[0] / 2)
    right  = robot_pos[1] + (ROBOT_SIZE[1] / 2)

    return top, left, bottom, right


def to_wpilib_coords(cv2_pos):
    return cv2_pos[1], FIELD_HEIGHT - cv2_pos[0], cv2_pos[2]
