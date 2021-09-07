from common.field_constants import FIELD_HEIGHT, ROBOT_SIZE


def pose3d_to_frame2d(object_pose, scaled_values):
    pose3d = object_pose.copy()
    pose3d[1] = FIELD_HEIGHT - pose3d[1]
    pose = list(int(l * r) for l, r in zip(scaled_values, object_pose[:2]))
    # pose.append(object_pose[3])

    return pose


def object_pose_to_robot_pose(object_pose, camera_transform):
    robot_pose = object_pose
    return robot_pose


def robot_pose_to_frame2d_points(robot_pos, scaled_values):
    left   = robot_pos[0] - (ROBOT_SIZE[1] / 2)
    top    = FIELD_HEIGHT - robot_pos[2] - (ROBOT_SIZE[0] / 2)
    right  = robot_pos[0] + (ROBOT_SIZE[1] / 2)
    bottom = FIELD_HEIGHT - robot_pos[2] + (ROBOT_SIZE[0] / 2)

    left_top = pose3d_to_frame2d([left, top, 0, robot_pos[2]], scaled_values)
    right_bottom = pose3d_to_frame2d([right, bottom, 0, robot_pos[2]], scaled_values)

    return left_top, right_bottom


def frame2d_to_field2d(cv2_pos):
    return cv2_pos[1], cv2_pos[0], cv2_pos[2]


def pose3d_to_field2d(pose3d):
    return pose3d[2], pose3d[0]