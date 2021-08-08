import contextlib
import logging
import uuid
from pathlib import Path

import blobconverter
import cv2
import depthai as dai

from config import *
from imutils.video import FPS

log = logging.getLogger(__name__)


def create_pipeline(model_name):
    log.info("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()

    xoutRgb = pipeline.createXLinkOut()
    camRgb.preview.link(xoutRgb.input)
    xoutNN = pipeline.createXLinkOut()

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
    blob_path = model_dir / Path(model_name).with_suffix(f".blob")

    config_path = model_dir / Path(model_name).with_suffix(f".json")
    nn_config = NNConfig(config_path)
    labels = nn_config.labels

    spatialDetectionNetwork.setBlobPath(str(blob_path))
    spatialDetectionNetwork.setConfidenceThreshold(nn_config.confidence)
    spatialDetectionNetwork.setNumClasses(nn_config.metadata["classes"])
    spatialDetectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
    spatialDetectionNetwork.setAnchors(nn_config.metadata["anchors"])
    spatialDetectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
    spatialDetectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")

    # Linking
    camRgb.preview.link(spatialDetectionNetwork.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    log.info("Pipeline created.")

    return pipeline, labels


def init_devices(device_list, pipeline):
    devices = {}
    preview_queues = {}
    detection_queues = {}

    for device_name, device_params in device_list.items():
        found_device, device_info = dai.Device.getDeviceByMxId(device_params['id'])

        if not found_device:
            log.warning("Device {}: Not Found".format(device_name))
            continue

        device = dai.Device(pipeline, device_info)
        device.startPipeline()

        devices[device_params['name']] = device
        preview_queues[device_params['name']] = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detection_queues[device_params['name']] = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    return devices, preview_queues, detection_queues


def capture(previewQueue, detectionNNQueue, labels):
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

            if DEBUG:
                cv2.rectangle(frame, (detection['x_min'], detection['y_min']), (detection['x_max'], detection['y_max']), (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(detection['depth_x'], 2)), (detection['x_min'], detection['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "y: {}".format(round(detection['depth_y'], 2)), (detection['x_min'], detection['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "conf: {}".format(round(detection['confidence'], 2)), (detection['x_min'], detection['y_min'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "label: {}".format(labels[detection['label']], 1), (detection['x_min'], detection['y_min'] + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        return frame, bboxes


def parse_frame(frame, bboxes, valid_labels):
    for bbox in bboxes:
        pass

    return results


def stream_frame():
    pass
