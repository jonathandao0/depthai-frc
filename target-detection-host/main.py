#!/usr/bin/env python3

import argparse
import cv2
import depthai as dai
import socket

import goal_depthai_utils
import object_depthai_utils
import logging
import target_detection

from common.mjpeg_stream import MjpegStream
from networktables.util import NetworkTables
from utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:

    def __init__(self):
        log.info("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.info(f"{device.getMxId()} {device.state}")

        self.init_networktables()

        ip_address = socket.gethostbyname(socket.gethostname())
        port1 = 4201
        port2 = 4202

        self.device_list = {"OAK-1": {
            'name': "OAK-1",
            # 'id': "14442C10C14F47D700",
            'id': "14442C1011043ED700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port1),
            'nt_tab': NetworkTables.getTable("OAK-1")
        }, "OAK-2": {
            'name': "OAK-2",
            'id': "14442C10C14F47D700",
            # 'id': "14442C1011043ED700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port2),
            'nt_tab': NetworkTables.getTable("OAK-2")
        }}

        self.goal_pipeline, self.goal_labels = goal_depthai_utils.create_pipeline("infiniteRecharge2021")
        self.object_pipeline, self.object_labels = object_depthai_utils.create_pipeline("infiniteRecharge2021")

        self.ip_address = socket.gethostname()
        self.oak_1_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port1)
        self.oak_2_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port2)

    def parse_goal_frame(self, frame, bboxes, edgeFrame):
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        nt_tab = self.device_list['OAK-1']['nt_tab']
        nt_tab.putNumber("tv", 1 if len(bboxes) > 0 else 0)

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]
            if target_label not in valid_labels:
                continue

            edgeFrame, target_x, target_y = target_detection.find_largest_contour(edgeFrame, bbox)

            if target_x == -999 or target_y == -999:
                log.error("Error: Could not find target contour")
                continue

            angle_offset = (target_x - (goal_depthai_utils.NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1080

            log.info("Found target '{}'\tX Angle Offset: {}".format(target_label, angle_offset))

            nt_tab.putString("target_label", target_label)
            nt_tab.putNumber("tx", angle_offset)

            # cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
            #               (0, 255, 0), 2)
            # cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 30),
            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            # cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            # cv2.putText(frame, "angle: {}".format(round(angle_offset, 2)), (bbox['x_min'], bbox['y_min'] + 70),
            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            # cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 90),
            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            # cv2.putText(frame, "label: {}".format(self.goal_labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 110),
            #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

            cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128), thickness=-1)

        self.oak_1_stream.sendFrame(edgeFrame)

    def parse_object_frame(self, frame, bboxes):
        valid_labels = ['power_cell']

        nt_tab = self.device_list['OAK-2']['nt_tab']
        power_cell_counter = 0
        for bbox in bboxes:
            target_label = self.object_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            power_cell_counter += 1

        box_color = (255, 255, 0)
        if power_cell_counter >= 5:
            box_color = (0, 255, 0)
        elif power_cell_counter < 3:
            box_color = (255, 0, 0)

        for bbox in bboxes:
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), box_color, 2)

        nt_tab.putNumber("powercells", power_cell_counter)
        nt_tab.putBoolean("indexer_full", power_cell_counter >= 5)

        self.oak_2_stream.sendFrame(frame)

    def init_networktables(self):
        NetworkTables.startClientTeam(4201)

        if not NetworkTables.isConnected():
            log.info("Could not connect to team client. Trying other addresses...")
            NetworkTables.startClient([
                '10.42.1.2',
                '127.0.0.1',
                '10.0.0.2',
                '192.168.100.108'
            ])

        if NetworkTables.isConnected():
            log.info("NT Connected to {}".format(NetworkTables.getRemoteAddress()))
            return True
        else:
            log.error("Could not connect to NetworkTables. Restarting server...")
            return False

    def run(self):
        log.info("Setup complete, parsing frames...")

        try:
            found_1, device_info_1 = dai.Device.getDeviceByMxId(self.device_list['OAK-1']['id'])
            self.device_list['OAK-1']['nt_tab'].putBoolean("OAK-1 Status", found_1)

            if found_1:
                self.device_list['OAK-1']['nt_tab'].putString("OAK-1 Stream", self.device_list['OAK-1']['stream_address'])
                for frame, bboxes, edgeFrame in goal_depthai_utils.capture(device_info_1):
                    self.parse_goal_frame(frame, bboxes, edgeFrame)

            found_2, device_info_2 = dai.Device.getDeviceByMxId(self.device_list['OAK-2']['id'])
            self.device_list['OAK-2']['nt_tab'].putBoolean("OAK-2 Status", found_2)

            if found_2:
                self.device_list['OAK-1']['nt_tab'].putString("OAK-2 Stream", self.device_list['OAK-2']['stream_address'])
                for frame, bboxes in goal_depthai_utils.capture(device_info_2):
                    self.parse_object_frame(frame, bboxes)

        finally:
            log.info("Exiting Program...")


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_goal_frame(self, frame, bboxes, edgeFrame):
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        nt_tab = self.device_list['OAK-1']['nt_tab']
        nt_tab.putNumber("tv", 1 if len(bboxes) > 0 else 0)

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            edgeFrame, target_x, target_y = target_detection.find_largest_contour(edgeFrame, bbox)

            angle_offset = (target_x - (goal_depthai_utils.NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1080

            log.info("Found target '{}'\tX Angle Offset: {}".format(target_label, angle_offset))

            nt_tab.putString("target_label", target_label)
            nt_tab.putNumber("tx", angle_offset)

            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (0, 255, 0), 2)
            cv2.putText(frame, "x: {}".format(round(target_x, 2)), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "label: {}".format(self.goal_labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

            cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

            cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128), thickness=-1)

        cv2.imshow("OAK-1", frame)
        cv2.imshow("OAK-1 Edge", edgeFrame)

        self.oak_1_stream.sendFrame(edgeFrame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
