import logging
import math

import cv2
import numpy as np

from config import MODEL_NAME, DEBUG, NN_IMG_SIZE
from depthai_utils import DepthAI, DepthAIDebug
from distance import DistanceCalculations, DistanceCalculationsDebug
from field_constants import *
from networktables import NetworkTables

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI
    distance_class = DistanceCalculations
    network_tables = NetworkTables.initialize(server="localhost")
    smartdashboard = NetworkTables.getTable("Depthai")

    robot_pose = None
    has_targets = False

    def __init__(self):
        self.depthai = self.depthai_class(MODEL_NAME)
        self.distance = self.distance_class()

        self.robot_pose = (FIELD_HEIGHT / 2, FIELD_WIDTH / 2, 0)

    def parse_frame(self, frame, results):
        distance_results = self.distance.parse_frame(frame, results)

        for result in results:
            self.has_targets = True

            if result['label'] == 5 or result['label'] == 4:
                self.robot_pose = (RED_POWER_PORT[1] - result['depth_x'], RED_POWER_PORT[0] + result['depth_z'], RED_POWER_PORT[2] - 180)
            elif result['label'] == 0 or result['label'] == 1:
                self.robot_pose = (BLUE_POWER_PORT[1] - result['depth_x'], BLUE_POWER_PORT[0] - result['depth_z'], BLUE_POWER_PORT[2] - 180)
            elif result['label'] == 3:
                self.robot_pose = (RED_LOADING_BAY[1] - result['depth_x'], RED_LOADING_BAY[0] - result['depth_z'], RED_LOADING_BAY[2] - 180)
            # if result['label'] == 6:
            #     self.robot_position = (BLUE_LOADING_BAY[1] - result['depth_x'], BLUE_LOADING_BAY[0] - result['depth_z'], BLUE_LOADING_BAY[2] - 180))

        if len(results) == 0:
            self.robot_pose = (-99, -99, 0)
            self.has_targets = False

        print("Robot Position: {}".format(self.robot_pose))
        self.smartdashboard.putBoolean("has_targets", self.has_targets)
        self.smartdashboard.putNumberArray("robot_pose", to_wpilib_coords(self.robot_pose))

        return distance_results

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results in self.depthai.capture():
                self.parse_frame(frame, results)

        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug
    distance_class = DistanceCalculationsDebug
    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7

    scaled_values = None
    red_power_port = None
    blue_power_port = None
    red_loading_bay = None
    blue_loading_bay = None
    red_station_1 = None
    red_station_2 = None
    red_station_3 = None
    blue_station_1 = None
    blue_station_2 = None
    blue_station_3 = None

    def __init__(self):
        super().__init__()
        self.field_frame = self.make_field_image()

    def make_field_image(self):
        fov = 68.7938
        min_distance = 0.827

        frame = cv2.imread("resources/images/frc2020fieldOfficial.png")
        height, width = frame.shape[:2]

        self.scaled_values = (width / FIELD_WIDTH, height / FIELD_HEIGHT)

        red_power_port = tuple(int(l * r) for l, r in zip(self.scaled_values, RED_POWER_PORT))
        blue_power_port = tuple(int(l * r) for l, r in zip(self.scaled_values, BLUE_POWER_PORT))
        red_loading_bay = tuple(int(l * r) for l, r in zip(self.scaled_values, RED_LOADING_BAY))
        blue_loading_bay = tuple(int(l * r) for l, r in zip(self.scaled_values, BLUE_LOADING_BAY))

        red_station_1 = tuple(int(l * r) for l, r in zip(self.scaled_values, RED_STATION_1))
        red_station_2 = tuple(int(l * r) for l, r in zip(self.scaled_values, RED_STATION_2))
        red_station_3 = tuple(int(l * r) for l, r in zip(self.scaled_values, RED_STATION_3))

        blue_station_1 = tuple(int(l * r) for l, r in zip(self.scaled_values, BLUE_STATION_1))
        blue_station_2 = tuple(int(l * r) for l, r in zip(self.scaled_values, BLUE_STATION_2))
        blue_station_3 = tuple(int(l * r) for l, r in zip(self.scaled_values, BLUE_STATION_3))

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

    def calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.field_frame.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.field_frame.shape[1])
        return int(bottom_x), int(top_x)

    def calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.field_frame.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.field_frame.shape[0])
        return int(bottom_z), int(top_z)

    def draw_robot(self, frame):
        r_t, r_l, r_b, r_r = robot_position_to_frame_coords(tuple(self.robot_pose[:2]))
        r_left_top = (tuple(int(l * r) for l, r in zip(self.scaled_values, (r_l, r_t))))
        r_right_bottom = tuple(int(l * r) for l, r in zip(self.scaled_values, (r_r, r_b)))

        cv2.rectangle(frame, r_left_top, r_right_bottom, (0, 255, 0), -1)

    def parse_frame(self, frame, results):
        distance_results = super().parse_frame(frame, results)

        field_frame = self.field_frame.copy()
        too_close_ids = []
        for result in distance_results:
            if result['dangerous']:
                left, right = self.calc_x(result['detection1']['depth_x'])
                top, bottom = self.calc_z(result['detection1']['depth_z'])
                cv2.rectangle(field_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection1']['id'])
                left, right = self.calc_x(result['detection2']['depth_x'])
                top, bottom = self.calc_z(result['detection2']['depth_z'])
                cv2.rectangle(field_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection2']['id'])

        self.draw_robot(field_frame)

        cv2.imshow("Field", field_frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    # if DEBUG:
    #     log.info("Setting up debug run...")
    #     MainDebug().run()
    # else:
        log.info("Setting up non-debug run...")
        Main().run()