import contextlib
import logging
import uuid
from pathlib import Path

import blobconverter
import cv2
import depthai as dai

from common.config import *
from imutils.video import FPS

log = logging.getLogger(__name__)


def create_pipeline(model_name):
    global pipeline
    log.info("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.createYoloDetectionNetwork()

    xoutRgb = pipeline.createXLinkOut()
    xoutNN = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(60)

    model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
    blob_path = model_dir / Path(model_name).with_suffix(f".blob")

    config_path = model_dir / Path(model_name).with_suffix(f".json")
    nn_config = NNConfig(config_path)
    labels = nn_config.labels

    detectionNetwork.setBlobPath(str(blob_path))
    detectionNetwork.setConfidenceThreshold(nn_config.confidence)
    detectionNetwork.setNumClasses(nn_config.metadata["classes"])
    detectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
    detectionNetwork.setAnchors(nn_config.metadata["anchors"])
    detectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
    detectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking
    camRgb.preview.link(detectionNetwork.input)
    # detectionNetwork.passthrough.link(xoutRgb.input)
    camRgb.preview.link(xoutRgb.input)
    detectionNetwork.out.link(xoutNN.input)

    log.info("Pipeline created.")

    return pipeline, labels


def init_devices(device_list, pipeline):
    devices = {}

    for device_name, device_params in device_list.items():
        found_device, device_info = dai.Device.getDeviceByMxId(device_params['id'])

        if not found_device:
            log.warning("Device {}: Not Found".format(device_name))
            continue

        device = dai.Device(pipeline, device_info)
        device.startPipeline()

        devices[device_params['name']] = device

    return devices


def capture(device_info):
    with dai.Device(pipeline, device_info) as device:
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        while True:
            frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            bboxes = []
            height = frame.shape[0]
            width  = frame.shape[1]
            for detection in detections:
                bboxes.append({
                    'id': uuid.uuid4(),
                    'label': detection.label,
                    'confidence': detection.confidence,
                    'x_min': int(detection.xmin * width),
                    'x_mid': int((detection.xmax - detection.xmin) * width),
                    'x_max': int(detection.xmax * width),
                    'y_min': int(detection.ymin * height),
                    'y_mid': int((detection.ymax - detection.ymin) * height),
                    'y_max': int(detection.ymax * height),
                })

            yield frame, bboxes


def del_pipeline():
    del pipeline
