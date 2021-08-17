import cv2
from common.field_constants import FIELD_HEIGHT, FIELD_WIDTH, pose3d_to_frame_position, LANDMARKS, \
    robot_pose_to_frame_position


class Field2dWindow:

    def init_field_image(self, frame):
        fov = 68.7938
        min_distance = 0.827

        red_power_port = pose3d_to_frame_position(LANDMARKS['red_upper_power_port_sandbox']['pose'], self.scaled_values)
        blue_power_port = pose3d_to_frame_position(LANDMARKS['blue_upper_power_port_sandbox']['pose'], self.scaled_values)
        red_loading_bay = pose3d_to_frame_position(LANDMARKS['red_loading_bay']['pose'], self.scaled_values)
        blue_loading_bay = pose3d_to_frame_position(LANDMARKS['blue_loading_bay']['pose'], self.scaled_values)

        red_station_1 = pose3d_to_frame_position(LANDMARKS['red_station_1']['pose'], self.scaled_values)
        red_station_2 = pose3d_to_frame_position(LANDMARKS['red_station_2']['pose'], self.scaled_values)
        red_station_3 = pose3d_to_frame_position(LANDMARKS['red_station_3']['pose'], self.scaled_values)

        blue_station_1 = pose3d_to_frame_position(LANDMARKS['blue_station_1']['pose'], self.scaled_values)
        blue_station_2 = pose3d_to_frame_position(LANDMARKS['blue_station_2']['pose'], self.scaled_values)
        blue_station_3 = pose3d_to_frame_position(LANDMARKS['blue_station_3']['pose'], self.scaled_values)

        cv2.circle(frame, red_power_port, 5, (0, 0, 255))
        cv2.circle(frame, blue_power_port, 5, (255, 0, 0))
        cv2.circle(frame, red_loading_bay, 5, (0, 0, 255))
        cv2.circle(frame, blue_loading_bay, 5, (255, 0, 0))

        cv2.circle(frame, red_station_1, 5, (0, 0, 255))
        cv2.circle(frame, red_station_2, 5, (0, 0, 255))
        cv2.circle(frame, red_station_3, 5, (0, 0, 255))

        cv2.circle(frame, blue_station_1, 5, (255, 0, 0))
        cv2.circle(frame, blue_station_2, 5, (255, 0, 0))
        cv2.circle(frame, blue_station_3, 5, (255, 0, 0))

        return frame

    def draw_robot_frame(self, robot_pose3d):
        output_frame = self.field_frame.copy()
        try:
            r_left_top, r_right_bottom = robot_pose_to_frame_position(robot_pose3d, self.scaled_values)

            cv2.rectangle(output_frame, r_left_top, r_right_bottom, (0, 255, 0), -1)
        except Exception as e:
            pass

        return output_frame

    def __init__(self):
        frame = cv2.imread("../resources/images/frc2020fieldOfficial.png")
        height, width = frame.shape[:2]

        self.scaled_values = (width / FIELD_WIDTH, height / FIELD_HEIGHT)
        self.field_frame = self.init_field_image(frame)