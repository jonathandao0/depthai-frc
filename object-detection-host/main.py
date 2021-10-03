#!/usr/bin/env python3

import argparse
import threading

import cv2
import depthai as dai
import socket

import goal_depthai_utils
import object_depthai_utils
import logging
import target_detection

from common.mjpeg_stream import MjpegStream
from networktables.util import NetworkTables

from common.utils import FPSHandler

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

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))

        ip_address = s.getsockname()[0]
        port1 = 4201
        port2 = 4202

        self.device_list = {"OAK-1": {
            'name': "OAK-1",
            'id': "14442C10C14F47D700",
            # 'id': "14442C1011043ED700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port1),
            'nt_tab': NetworkTables.getTable("OAK-1")
        }, "OAK-2": {
            'name': "OAK-2",
            # 'id': "14442C10C14F47D700",
            'id': "14442C1011043ED700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port2),
            'nt_tab': NetworkTables.getTable("OAK-2")
        }}

        self.goal_pipeline, self.goal_labels = goal_depthai_utils.create_pipeline("infiniteRecharge2021")
        self.object_pipeline, self.object_labels = object_depthai_utils.create_pipeline("infiniteRecharge2021")

        self.oak_1_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port1)
        self.oak_2_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port2)

    def parse_goal_frame(self, edgeFrame, bboxes):
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        nt_tab = self.device_list['OAK-1']['nt_tab']

        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]
            if target_label not in valid_labels:
                continue

            edgeFrame, target_x, target_y = target_detection.find_largest_contour(edgeFrame, bbox)

            if target_x == -999 or target_y == -999:
                log.error("Error: Could not find target contour")
                continue

            angle_offset = ((goal_depthai_utils.NN_IMG_SIZE / 2.0) - target_x) * 68.7938003540039 / 1080

            if abs(angle_offset) > 30:
                log.info("Invalid angle offset. Setting it to 0")
                nt_tab.putNumber("tv", 0)
                angle_offset = 0
            else:
                log.info("Found target '{}'\tX Angle Offset: {}".format(target_label, angle_offset))
                nt_tab.putNumber("tv", 1)

            nt_tab.putString("target_label", target_label)
            nt_tab.putNumber("tx", angle_offset)

            cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

            cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128), thickness=-1)

            bbox['target_x'] = target_x
            bbox['target_y'] = target_y
            bbox['angle_offset'] = angle_offset

        fps = self.device_list['OAK-1']['fps_handler']
        fps.next_iter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        self.oak_1_stream.send_frame(edgeFrame)

        return edgeFrame, bboxes

    def parse_object_frame(self, frame, bboxes):
        valid_labels = ['power_cell']

        nt_tab = self.device_list['OAK-2']['nt_tab']
        power_cell_counter = 0
        for bbox in bboxes:
            target_label = self.object_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            power_cell_counter += 1

        box_color = (0, 150, 150)
        if power_cell_counter >= 5:
            box_color = (0, 255, 0)
        elif power_cell_counter < 3:
            box_color = (0, 0, 255)

        for bbox in bboxes:
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), box_color, 2)

        nt_tab.putNumber("powercells", power_cell_counter)
        nt_tab.putBoolean("indexer_full", power_cell_counter >= 5)

        self.oak_2_stream.send_frame(frame)

        return frame, bboxes

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
                th1 = threading.Thread(target=self.run_goal_detection, args=(device_info_1,))
                th1.start()

            found_2, device_info_2 = dai.Device.getDeviceByMxId(self.device_list['OAK-2']['id'])
            self.device_list['OAK-2']['nt_tab'].putBoolean("OAK-2 Status", found_2)

            if found_2:
                th2 = threading.Thread(target=self.run_object_detection, args=(device_info_2,))
                th2.start()

        finally:
            log.info("Exiting Program...")

    def run_goal_detection(self, device_info):
        self.device_list['OAK-1']['nt_tab'].putString("OAK-1 Stream", self.device_list['OAK-1']['stream_address'])
        for edgeFrame, bboxes in goal_depthai_utils.capture(device_info):
            self.parse_goal_frame(edgeFrame, bboxes)

    def run_object_detection(self, device_info):
        self.device_list['OAK-1']['nt_tab'].putString("OAK-2 Stream", self.device_list['OAK-2']['stream_address'])
        for frame, bboxes in object_depthai_utils.capture(device_info):
            self.parse_object_frame(frame, bboxes)


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_goal_frame(self, frame, bboxes):
        edgeFrame, bboxes = super().parse_goal_frame(frame, bboxes)
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            if 'target_x' not in bbox:
                continue

            target_x = bbox['target_x']
            angle_offset = bbox['angle_offset']

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

        cv2.imshow("OAK-1 Edge", edgeFrame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()

    def parse_object_frame(self, frame, bboxes):
        frame, bboxes = super().parse_object_frame(frame, bboxes)

        for bbox in bboxes:
            cv2.putText(frame, "id: {}".format(bbox['id']), (bbox['x_min'], bbox['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "status: {}".format(bbox['status']), (bbox['x_min'], bbox['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.imshow("OAK-1 Objects", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
