import logging
import uuid
from pathlib import Path

import blobconverter
import cv2
import depthai as dai

from config import *
from imutils.video import FPS

log = logging.getLogger(__name__)


class DepthAI:
    nn_config = None
    labels = None

    def create_pipeline(self, model_name):
        log.info("Creating DepthAI pipeline...")

        # out_depth = False  # Disparity by default
        # out_rectified = True  # Output and display rectified streams
        # lrcheck = True  # Better handling for occlusions
        # extended = False  # Closer-in minimum depth, disparity range is doubled
        # subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
        # # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
        # median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Define sources and outputs
        camRgb = pipeline.createColorCamera()
        spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
        monoLeft = pipeline.createMonoCamera()
        monoRight = pipeline.createMonoCamera()
        stereo = pipeline.createStereoDepth()

        xoutRgb = pipeline.createXLinkOut()
        camRgb.preview.link(xoutRgb.input)
        xoutNN = pipeline.createXLinkOut()

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")

        # Properties
        camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        # stereo.setOutputDepth(out_depth)
        # stereo.setOutputRectified(out_rectified)
        stereo.setConfidenceThreshold(255)
        stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
        # stereo.setMedianFilter(median) # KERNEL_7x7 default
        # stereo.setLeftRightCheck(lrcheck)
        # stereo.setExtendedDisparity(extended)
        # stereo.setSubpixel(subpixel)

        model_dir = Path(__file__).parent / Path(f"resources/nn/") / MODEL_NAME
        blob_path = model_dir / Path(MODEL_NAME).with_suffix(f".blob")

        config_path = model_dir / Path(MODEL_NAME).with_suffix(f".json")
        self.nn_config = NNConfig(config_path)
        self.labels = self.nn_config.labels

        spatialDetectionNetwork.setBlobPath(str(blob_path))
        spatialDetectionNetwork.setConfidenceThreshold(self.nn_config.confidence)
        spatialDetectionNetwork.setNumClasses(self.nn_config.metadata["classes"])
        spatialDetectionNetwork.setCoordinateSize(self.nn_config.metadata["coordinates"])
        spatialDetectionNetwork.setAnchors(self.nn_config.metadata["anchors"])
        spatialDetectionNetwork.setAnchorMasks(self.nn_config.metadata["anchor_masks"])
        spatialDetectionNetwork.setIouThreshold(self.nn_config.metadata["iou_threshold"])
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)


        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        log.info("Pipeline created.")
        return pipeline

    def __init__(self, model_name):
        self.pipeline = self.create_pipeline(model_name)
        self.detections = []

    def capture(self):
        with dai.Device(self.pipeline) as device:
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

            while True:
                frame = previewQueue.get().getCvFrame()
                inDet = detectionNNQueue.tryGet()

                if inDet is not None:
                    self.detections = inDet.detections

                bboxes = []
                height = frame.shape[0]
                width  = frame.shape[1]
                for detection in self.detections:
                    bboxes.append({
                        'id': uuid.uuid4(),
                        'label': detection.label,
                        'confidence': detection.confidence,
                        'x_min': int(detection.xmin * width),
                        'x_max': int(detection.xmax * width),
                        'y_min': int(detection.ymin * height),
                        'y_max': int(detection.ymax * height),
                        'depth_x': detection.spatialCoordinates.x / 1000,
                        'depth_y': detection.spatialCoordinates.y / 1000,
                        'depth_z': detection.spatialCoordinates.z / 1000,
                    })

                yield frame, bboxes

    def __del__(self):
        del self.pipeline


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def capture(self):
        for frame, detections in super().capture():
            self.fps.update()
            for detection in detections:
                cv2.rectangle(frame, (detection['x_min'], detection['y_min']), (detection['x_max'], detection['y_max']), (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(detection['depth_x'], 2)), (detection['x_min'], detection['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "y: {}".format(round(detection['depth_y'], 2)), (detection['x_min'], detection['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "z: {}".format(round(detection['depth_z'], 2)), (detection['x_min'], detection['y_min'] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "conf: {}".format(round(detection['confidence'], 2)), (detection['x_min'], detection['y_min'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "label: {}".format(self.labels[detection['label']], 1), (detection['x_min'], detection['y_min'] + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            yield frame, detections

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
