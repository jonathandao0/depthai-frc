#!/usr/bin/env python3

import argparse
import operator
import threading
from time import sleep

import cv2
import depthai as dai
import socket

from common.config import NN_IMG_SIZE
from pipelines import goal_edge_depth_detection, object_edge_detection
import logging
from common import target_finder

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

        self.device_list = {"OAK-D_Goal": {
            'name': "OAK-D_Goal",
            'id': "14442C10218CCCD200",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port1),
            'nt_tab': NetworkTables.getTable("OAK-D_Goal")
        }, "OAK-1_Intake": {
            'name': "OAK-1_Intake",
            # 'id': "14442C1011043ED700",
            'id': "14442C10C14F47D700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port2),
            'nt_tab': NetworkTables.getTable("OAK-1_Intake")
        }}

        self.goal_pipeline, self.goal_labels = goal_edge_depth_detection.create_pipeline("infiniteRecharge2021")
        self.intake_pipeline, self.intake_labels = object_edge_detection.create_pipeline("infiniteRecharge2021")

        self.oak_d_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port1, colorspace='BW', QUALITY=10)
        self.oak_1_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port2, colorspace='BW', QUALITY=10)

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        edgeFrame = cv2.threshold(edgeFrame, 20, 255, cv2.THRESH_TOZERO)[1]

        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        nt_tab = self.device_list['OAK-D_Goal']['nt_tab']

        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)
        else:
            for bbox in bboxes:
                target_label = self.goal_labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                edgeFrame, target_x, target_y = target_finder.find_largest_contour(edgeFrame, bbox)

                if target_x == -999 or target_y == -999:
                    log.error("Error: Could not find target contour")
                    continue

                angle_offset = (target_x - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920

                if abs(angle_offset) > 30:
                    log.info("Invalid angle offset. Setting it to 0")
                    nt_tab.putNumber("tv", 0)
                    angle_offset = 0
                else:
                    log.info("Found target '{}'\tX Angle Offset: {}".format(target_label, angle_offset))
                    nt_tab.putNumber("tv", 1)

                nt_tab.putString("target_label", target_label)
                nt_tab.putNumber("tx", angle_offset)
                nt_tab.putNumber("tz", bbox['depth_z'])

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

                cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128),
                           thickness=-1)

                bbox['target_x'] = target_x
                bbox['target_y'] = target_y
                bbox['angle_offset'] = angle_offset

        fps = self.device_list['OAK-D_Goal']['fps_handler']
        fps.next_iter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        self.oak_d_stream.send_frame(edgeFrame)

        return frame, edgeFrame, bboxes

    def parse_intake_frame(self, frame, edgeFrame, bboxes):
        edgeFrame = cv2.threshold(frame, 60, 255, cv2.THRESH_TOZERO)[1]

        valid_labels = ['power_cell']

        nt_tab = self.device_list['OAK-1_Intake']['nt_tab']

        filtered_bboxes = []
        for bbox in bboxes:
            if self.intake_labels[bbox['label']] in valid_labels:
                filtered_bboxes.append(bbox)

        filtered_bboxes.sort(key=operator.itemgetter('size'), reverse=True)

        if len(filtered_bboxes) == 0:
            nt_tab.putNumber("tv", 0)
            nt_tab.putNumberArray("ta", [0])
        else:
            nt_tab.putNumber("tv", 1)

            target_angles = []
            for bbox in filtered_bboxes:
                angle_offset = (bbox['x_mid'] - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

                target_angles.append(angle_offset)
                bbox['angle_offset'] = angle_offset

            nt_tab.putNumberArray("ta", target_angles)

        fps = self.device_list['OAK-1_Intake']['fps_handler']
        fps.next_iter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        self.oak_1_stream.send_frame(edgeFrame)

        return frame, edgeFrame, filtered_bboxes

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

        threadlist = []
        try:
            found_1, device_info_1 = dai.Device.getDeviceByMxId(self.device_list['OAK-D_Goal']['id'])
            self.device_list['OAK-D_Goal']['nt_tab'].putBoolean("OAK-D_Goal Status", found_1)

            if found_1:
                th1 = threading.Thread(target=self.run_goal_detection, args=(device_info_1,))
                th1.start()
                threadlist.append(th1)

            found_2, device_info_2 = dai.Device.getDeviceByMxId(self.device_list['OAK-1_Intake']['id'])
            self.device_list['OAK-1_Intake']['nt_tab'].putBoolean("OAK-1_Intake Status", found_2)

            if found_2:
                th2 = threading.Thread(target=self.run_intake_detection, args=(device_info_2,))
                th2.start()
                threadlist.append(th2)

            while True:
                for t in threadlist:
                    if not t.is_alive():
                        break
                sleep(10)
        finally:
            log.info("Exiting Program...")

    def run_goal_detection(self, device_info):
        self.device_list['OAK-D_Goal']['nt_tab'].putString("OAK-D_Goal Stream", self.device_list['OAK-D_Goal']['stream_address'])
        for frame, edgeFrame, bboxes in goal_edge_depth_detection.capture(device_info):
            self.parse_goal_frame(frame, edgeFrame, bboxes)

    def run_intake_detection(self, device_info):
        self.device_list['OAK-1_Intake']['nt_tab'].putString("OAK-1 Stream", self.device_list['OAK-1_Intake']['stream_address'])
        for frame, edgeFrame, bboxes in object_edge_detection.capture(device_info):
            self.parse_intake_frame(frame, edgeFrame, bboxes)


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        frame, edgeFrame, bboxes = super().parse_goal_frame(frame, edgeFrame, bboxes)
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            target_x = bbox['target_x'] if 'target_x' in bbox else 0
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            cv2.putText(edgeFrame, "x: {}".format(round(target_x, 2)), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "z: {}".format(round(bbox['depth_z'], 2)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "label: {}".format(self.goal_labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 130),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.imshow("OAK-D Goal Edge", edgeFrame)
        cv2.imshow("OAK-D Goal ", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()

    def parse_intake_frame(self, frame, edgeFrame,  bboxes):
        frame, edgeFrame, bboxes = super().parse_intake_frame(frame, edgeFrame, bboxes)

        for i, bbox in enumerate(bboxes):
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            frame_color = (0, 255, 0) if i == 0 else (0, 150, 150)

            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), frame_color, 2)
            cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "size: {}".format(round(bbox['size'], 3)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.imshow("OAK-1 Intake Edge", edgeFrame)
        cv2.imshow("OAK-1 Intake", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    log.info("Starting goal-depth-detection-host")
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
