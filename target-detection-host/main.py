import logging

import argparse

import cv2
import depthai
import depthai as dai
import target_detection

from networktables.util import NetworkTables

import depthai_utils
import contextlib

from common.config import MODEL_NAME
from common.mjpeg_stream import MjpegStream
from utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:
    network_tables = NetworkTables.initialize(server='localhost')

    def __init__(self):
        for device in dai.Device.getAllAvailableDevices():
            print(f"{device.getMxId()} {device.state}")

        self.device_list = {"OAK-1": {
            'name': "OAK-1",
            'id': "14442C10C14F47D700",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-1")
        }, "OAK-2": {
            'name': "OAK-2",
            'id': "14442C1091398FD000",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-2")
        }}

        self.pipeline, self.labels = depthai_utils.create_pipeline("infiniteRecharge2021")
        self.oak_1_stream = MjpegStream(1181)
        self.oak_2_stream = MjpegStream(1182)
        # self.devices = depthai_utils.init_devices(self.device_list, self.pipeline)

    def parse_frame(self, frame, bboxes, edgeFrame, device):
        device_name = device['name']

        if device_name == "OAK-1":
            valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

            nt_tab = self.device_list['OAK-1']['nt_tab']
            for bbox in bboxes:
                target_label = self.labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                target_x = target_detection.find_target_center(edgeFrame, bbox)

                angle_offset = (target_x - (depthai_utils.NN_IMG_SIZE / 2)) * 68.7938003540039 / 1080
                nt_tab.putString("Target", target_label)
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
                # cv2.putText(frame, "label: {}".format(self.labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 110),
                #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

                self.oak_1_stream.sendFrame(edgeFrame)
        elif device_name == "OAK-2":
            valid_labels = ['powercell']

    def run(self):
        log.info("Setup complete, parsing frames...")
        try:
            for device in self.device_list:
                device_id = self.device_list[device]['id']
                found, device_info = depthai.Device.getDeviceByMxId(device_id)

                for frame, bboxes, edgeFrame in depthai_utils.capture(device_info):
                    self.parse_frame(frame, bboxes, edgeFrame, self.device_list[device])

        finally:
            pass
            # for device in self.devices:
            #     device.close()


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_frame(self, frame, bboxes, edgeFrame, device):
        device_name = device['name']

        if device_name == "OAK-1":
            valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

            nt_tab = self.device_list['OAK-1']['nt_tab']
            for bbox in bboxes:
                target_label = self.labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                target_x = target_detection.find_target_center(edgeFrame, bbox)

                angle_offset = (target_x - (depthai_utils.NN_IMG_SIZE / 2)) * 68.7938003540039 / 1080
                nt_tab.putString("Target", target_label)
                nt_tab.putNumber("tx", angle_offset)

                cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "angle: {}".format(round(angle_offset, 2)), (bbox['x_min'], bbox['y_min'] + 70),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 90),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "label: {}".format(self.labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 110),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

            cv2.imshow("OAK-1", frame)
            cv2.imshow("OAK-1 Edge", edgeFrame)

        elif device_name == "OAK-2":
            valid_labels = ['powercell']

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
