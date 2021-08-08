import logging

import argparse
import depthai as dai
from networktables.util import NetworkTables

import depthai_utils
import contextlib

from config import MODEL_NAME
from utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:
    network_tables = NetworkTables.initialize(server='localhost')

    def __init__(self):
        for device in dai.Device.getAllAvailableDevices():
            print(f"{device.getMxId()} {device.state}")

        self.device_list = {"OAK-1": {
            'name': "OAK-1",
            'id': "14442C10218CCCD200",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-1")
        }, "OAK-2": {
            'name': "OAK-2",
            'id': "14442C1091398FD000",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-2")
        }}

        self.pipeline, self.labels = depthai_utils.create_pipeline(MODEL_NAME)
        self.devices, self.preview_queues, self.detection_queues = depthai_utils.init_devices(self.device_list, self.pipeline)

    def run(self):
        log.info("Setup complete, parsing frames...")
        # try:
        while True:
            for device_name, device in self.devices.items():
                frames, bboxes = depthai_utils.capture(self.preview_queues[device_name], self.detection_queues[device_name], self.labels)

                if device_name == "OAK-1":
                    valid_labels = ['red_upper_power_port_sandbox', 'blue_upper_power_port_sandbox']
                elif device_name == "OAK-2":
                    valid_labels = ['powercell']

                results = depthai_utils.parse_frame(frames, bboxes, valid_labels)

        # finally:
        #     for device in self.devices:
        #         device.close


class MainDebug(Main):

    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
