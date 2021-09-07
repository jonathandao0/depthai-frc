import json
import logging
import os
import sys

from concurrent_log_handler import ConcurrentRotatingFileHandler

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logfile = os.path.abspath("../camera.log")
# Rotate log after reaching 512K, keep 5 old copies.
rotateHandler = ConcurrentRotatingFileHandler(logfile, "a", 512*1024, 5)
rotateHandler.setFormatter(formatter)
root.addHandler(rotateHandler)
root.setLevel(logging.INFO)
root.info("Logging system initialized, kept in file {}...".format(logfile))

MODEL_NAME = "infiniteRecharge2020sandbox"
NN_IMG_SIZE = 416
DEBUG = os.getenv('DEBUG', 'true') not in ('false', '0')

DETECTION_PADDING = 5


class NNConfig:
    source_choices = ("color", "left", "right", "rectified_left", "rectified_right", "host")
    config = None
    nn_family = None
    handler = None
    labels = None
    input_size = None
    confidence = None
    metadata = None
    openvino_version = None
    output_format = "raw"
    blob_path = None
    count_label = None

    def __init__(self, config_path=None):
        global LABELS

        if config_path.exists():
            with config_path.open() as f:
                self.config = json.load(f)

                nn_config = self.config.get("nn_config", {})
                self.labels = self.config.get("mappings", {}).get("labels", None)
                self.nn_family = nn_config.get("NN_family", None)
                self.output_format = nn_config.get("output_format", "raw")
                self.metadata = nn_config.get("NN_specific_metadata", {})
                if "input_size" in nn_config:
                    self.input_size = tuple(map(int, nn_config.get("input_size").split('x')))

                self.confidence = self.metadata.get("confidence_threshold", nn_config.get("confidence_threshold", None))
