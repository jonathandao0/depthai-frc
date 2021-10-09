#!/usr/bin/env python3

import depthai as dai
import uuid

from common.config import *
from pathlib import Path

log = logging.getLogger(__name__)


def create_pipeline(model_name):
    global pipeline
    log.info("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.createYoloDetectionNetwork()
    objectTracker = pipeline.createObjectTracker()

    xoutRgb = pipeline.createXLinkOut()
    rgbControl = pipeline.createXLinkIn()
    xinRgb = pipeline.createXLinkIn()
    trackerOut = pipeline.createXLinkOut()


    xoutRgb.setStreamName("rgb")
    xinRgb.setStreamName("rgbCfg")
    rgbControl.setStreamName('rgbControl')
    trackerOut.setStreamName("tracks")

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
    blob_path = model_dir / Path(model_name).with_suffix(f".blob")

    config_path = model_dir / Path(model_name).with_suffix(f".json")
    nn_config = NNConfig(config_path)
    labels = nn_config.labels

    detectionNetwork.setBlobPath(str(blob_path))
    detectionNetwork.setConfidenceThreshold(nn_config.confidence)
    # detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(nn_config.metadata["classes"])
    detectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
    detectionNetwork.setAnchors(nn_config.metadata["anchors"])
    detectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
    detectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    objectTracker.setDetectionLabelsToTrack([3])  # track only power cells
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

    # Linking
    camRgb.preview.link(detectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    camRgb.video.link(objectTracker.inputTrackerFrame)
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    objectTracker.out.link(trackerOut.input)

    log.info("Pipeline created.")

    return pipeline, labels


def capture(device_info):
    with dai.Device(pipeline, device_info) as device:
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        tracksQueue = device.getOutputQueue(name="tracks", maxSize=4, blocking=False)

        controlQueue = device.getInputQueue('rgbControl')
        configQueue = device.getInputQueue('rgbCfg')

        while True:
            cfg = dai.CameraControl()
            cfg.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            cfg.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.OFF)
            cfg.setAutoExposureLock(True)
            cfg.setAutoExposureCompensation(-6)
            configQueue.send(cfg)

            frame = previewQueue.get().getCvFrame()
            inTracks = tracksQueue.get()

            trackletsData = inTracks.tracklets
            bboxes = []
            for track in trackletsData:
                roi = track.roi.denormalize(frame.shape[1], frame.shape[0])
                bboxes.append({
                    'id': track.id,
                    'label': track.label,
                    'status': track.status,
                    # 'confidence': track.confidence,
                    'x_min': int(roi.topLeft().x),
                    'x_mid': int((roi.bottomRight().x - roi.topLeft().x) / 2 + roi.topLeft().x),
                    'x_max': int(roi.bottomRight().x),
                    'y_min': int(roi.topLeft().y),
                    'y_mid': int((roi.bottomRight().y - roi.topLeft().y) / 2 + roi.topLeft().y),
                    'y_max': int(roi.bottomRight().y),
                    'size': (roi.bottomRight().x - roi.topLeft().x) * (roi.bottomRight().y - roi.topLeft().y)
                })

            yield frame, bboxes


def del_pipeline():
    del pipeline
