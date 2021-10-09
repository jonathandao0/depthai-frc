#!/usr/bin/env python3

import argparse
import operator
import threading

import cv2
import depthai as dai
import socket

from common.config import NN_IMG_SIZE
from pipelines import object_detection, object_tracker_detection
import logging

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
            'nt_tab': NetworkTables.getTable("OAK-1_Intake")
        }, "OAK-2": {
            'name': "OAK-2",
            # 'id': "14442C10C14F47D700",
            'id': "14442C1011043ED700",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port2),
            'nt_tab': NetworkTables.getTable("OAK-2_Indexer")
        }}

        self.intake_pipeline, self.intake_labels = object_detection.create_pipeline("infiniteRecharge2021")
        self.object_pipeline, self.object_labels = object_tracker_detection.create_pipeline("infiniteRecharge2021")

        self.oak_1_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port1)
        self.oak_2_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port2)

    def parse_intake_frame(self, frame, bboxes):
        valid_labels = ['power_cell']

        nt_tab = self.device_list['OAK-1']['nt_tab']

        filtered_bboxes = []
        for bbox in bboxes:
            if self.intake_labels[bbox['label']] in valid_labels:
                filtered_bboxes.append(bbox)

        filtered_bboxes.sort(key=operator.itemgetter('size'), reverse=True)

        if len(filtered_bboxes) == 0:
            nt_tab.putNumber("tv", 0)
        else:
            nt_tab.putNumber("tv", 1)

            target_angles = []
            for bbox in filtered_bboxes:
                angle_offset = ((NN_IMG_SIZE / 2.0) - bbox['x_mid']) * 68.7938003540039 / 1080

                cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 255, 255), 2)

                target_angles.append(angle_offset)
                bbox['angle_offset'] = angle_offset

            nt_tab.putNumberArray("ta", target_angles)

        fps = self.device_list['OAK-1']['fps_handler']
        fps.next_iter()
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        self.oak_1_stream.send_frame(frame)

        return frame, filtered_bboxes

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

        fps = self.device_list['OAK-2']['fps_handler']
        fps.next_iter()
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

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
                th1 = threading.Thread(target=self.run_intake_detection, args=(device_info_1,))
                th1.start()

            found_2, device_info_2 = dai.Device.getDeviceByMxId(self.device_list['OAK-2']['id'])
            self.device_list['OAK-2']['nt_tab'].putBoolean("OAK-2 Status", found_2)

            if found_2:
                th2 = threading.Thread(target=self.run_object_detection, args=(device_info_2,))
                th2.start()

        finally:
            log.info("Exiting Program...")

    def run_intake_detection(self, device_info):
        self.device_list['OAK-1']['nt_tab'].putString("OAK-1 Stream", self.device_list['OAK-1']['stream_address'])
        for edgeFrame, bboxes in object_detection.capture(device_info):
            self.parse_intake_frame(edgeFrame, bboxes)

    def run_object_detection(self, device_info):
        self.device_list['OAK-1']['nt_tab'].putString("OAK-2 Stream", self.device_list['OAK-2']['stream_address'])
        for frame, bboxes in object_tracker_detection.capture(device_info):
            self.parse_object_frame(frame, bboxes)


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_intake_frame(self, frame, bboxes):
        frame, bboxes = super().parse_intake_frame(frame, bboxes)

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

        cv2.imshow("OAK-1 Intake", frame)

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