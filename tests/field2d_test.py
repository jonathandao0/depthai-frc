import cv2
import math
import numpy as np

from common import image_processing
from common.frame2dwindow import Frame2dWindow
from common.field_constants import FIELD_WIDTH, FIELD_HEIGHT, LANDMARKS


def main():
    field2d = Frame2dWindow()

    center_robot_pose3d = [FIELD_WIDTH / 2, FIELD_HEIGHT / 2, 0, 0]

    x = center_robot_pose3d[0]
    y = center_robot_pose3d[1]
    z = center_robot_pose3d[2]
    w = center_robot_pose3d[3]

    landmark_pos = LANDMARKS['red_upper_power_port']['pose']
    object_pos = np.array([landmark_pos[0], landmark_pos[2], landmark_pos[1]]).reshape((3, 1))

    tVec = np.array([[-0.20776848], [ 0.62778233], [ 0.75259799]])
    rVec = np.array([[ 1.07325922], [-0.38095453], [-2.6192035 ]])
    rMat = cv2.Rodrigues(rVec)[0]

    camera_pos = np.matmul(rMat.T, object_pos - tVec)

    x_pos = camera_pos[2][0]
    y_pos = -camera_pos[0][0]
    z_pos = camera_pos[1][0]

    yaw, pitch, roll = image_processing.decomposeYawPitchRoll(rMat.T)
    rotation_deg = 57.2958 * -yaw

    robot_pose3d = [x_pos, y_pos, z_pos, rotation_deg[0]]

    while True:
        print("Robot Pose: {}".format(robot_pose3d))
        field_frame = field2d.draw_robot(robot_pose3d)
        cv2.imshow("Field", field_frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    main()
