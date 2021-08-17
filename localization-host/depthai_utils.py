import uuid
from pathlib import Path

import cv2
import depthai as dai

from common.config import *
from imutils.video import FPS

from common.feature_tracker import FeatureTrackerDebug, FeatureTracker
from common.image_processing import SIFT_PARAMS

log = logging.getLogger(__name__)

LABELS = []


def create_pipeline(model_name):
    global pipeline
    global LABELS
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
    featureTrackerLeft = pipeline.createFeatureTracker()
    featureTrackerRight = pipeline.createFeatureTracker()
    stereo = pipeline.createStereoDepth()

    xoutRgb = pipeline.createXLinkOut()
    camRgb.preview.link(xoutRgb.input)
    xoutNN = pipeline.createXLinkOut()
    xoutPassthroughFrameLeft = pipeline.createXLinkOut()
    xoutTrackedFeaturesLeft = pipeline.createXLinkOut()
    xoutPassthroughFrameRight = pipeline.createXLinkOut()
    xoutTrackedFeaturesRight = pipeline.createXLinkOut()
    xinTrackedFeaturesConfig = pipeline.createXLinkIn()

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xoutPassthroughFrameLeft.setStreamName("passthroughFrameLeft")
    xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
    xoutPassthroughFrameRight.setStreamName("passthroughFrameRight")
    xoutTrackedFeaturesRight.setStreamName("trackedFeaturesRight")
    xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

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

    model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
    blob_path = model_dir / Path(model_name).with_suffix(f".blob")

    config_path = model_dir / Path(model_name).with_suffix(f".json")
    nn_config = NNConfig(config_path)
    LABELS = nn_config.labels

    spatialDetectionNetwork.setBlobPath(str(blob_path))
    spatialDetectionNetwork.setConfidenceThreshold(nn_config.confidence)
    spatialDetectionNetwork.setNumClasses(nn_config.metadata["classes"])
    spatialDetectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
    spatialDetectionNetwork.setAnchors(nn_config.metadata["anchors"])
    spatialDetectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
    spatialDetectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(50000)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    monoLeft.out.link(featureTrackerLeft.inputImage)
    featureTrackerLeft.passthroughInputImage.link(xoutPassthroughFrameLeft.input)
    featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)
    xinTrackedFeaturesConfig.out.link(featureTrackerLeft.inputConfig)

    monoRight.out.link(featureTrackerRight.inputImage)
    featureTrackerRight.passthroughInputImage.link(xoutPassthroughFrameRight.input)
    featureTrackerRight.outputFeatures.link(xoutTrackedFeaturesRight.input)
    xinTrackedFeaturesConfig.out.link(featureTrackerRight.inputConfig)

    numShaves = 2
    numMemorySlices = 2
    featureTrackerLeft.setHardwareResources(numShaves, numMemorySlices)
    featureTrackerRight.setHardwareResources(numShaves, numMemorySlices)

    featureTrackerConfig = featureTrackerRight.initialConfig.get()

    camRgb.preview.link(spatialDetectionNetwork.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    log.info("Pipeline created.")

    return pipeline, LABELS


def capture():
    with dai.Device(pipeline) as device:
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 8, False)
        outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)
        passthroughImageRightQueue = device.getOutputQueue("passthroughFrameRight", 8, False)
        outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 8, False)

        if DEBUG:
            leftFeatureTracker = FeatureTrackerDebug("Feature tracking duration (frames)", "Left")
            rightFeatureTracker = FeatureTrackerDebug("Feature tracking duration (frames)", "Right")
        else:
            leftFeatureTracker = FeatureTracker()
            rightFeatureTracker = FeatureTracker()

        while True:
            frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            bboxes = []
            featuredata = {}
            height = frame.shape[0]
            width  = frame.shape[1]
            for detection in detections:
                data = {
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
                }
                bboxes.append(data)

                if DEBUG:
                    cv2.rectangle(frame, (data['x_min'], data['y_min']), (data['x_max'], data['y_max']), (0, 255, 0), 2)
                    cv2.putText(frame, "x: {}".format(round(data['depth_x'], 2)), (data['x_min'], data['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, "y: {}".format(round(data['depth_y'], 2)), (data['x_min'], data['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, "z: {}".format(round(data['depth_z'], 2)), (data['x_min'], data['y_min'] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, "conf: {}".format(round(data['confidence'], 2)), (data['x_min'], data['y_min'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, "label: {}".format(LABELS[data['label']], 1), (data['x_min'], data['y_min'] + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

                target_sift_params = SIFT_PARAMS[LABELS[data['label']]]

                inPassthroughFrameLeft = passthroughImageLeftQueue.get()
                passthroughFrameLeft = inPassthroughFrameLeft.getFrame()
                leftFrame = cv2.cvtColor(passthroughFrameLeft, cv2.COLOR_GRAY2BGR)

                inPassthroughFrameRight = passthroughImageRightQueue.get()
                passthroughFrameRight = inPassthroughFrameRight.getFrame()
                rightFrame = cv2.cvtColor(passthroughFrameRight, cv2.COLOR_GRAY2BGR)

                trackedFeaturesLeft = outputFeaturesLeftQueue.get().trackedFeatures
                leftFeatureTracker.trackFeaturePath(trackedFeaturesLeft)
                left_good_matches, left_keypoints = leftFeatureTracker.matchRefImg(leftFrame, target_sift_params["descriptors"])

                trackedFeaturesRight = outputFeaturesRightQueue.get().trackedFeatures
                rightFeatureTracker.trackFeaturePath(trackedFeaturesRight)
                right_good_matches, right_keypoints = rightFeatureTracker.matchRefImg(rightFrame, target_sift_params["descriptors"])

                featuredata[data['id']] = {
                    'leftFrame': leftFrame,
                    'left_good_matches': left_good_matches,
                    'left_keypoints': left_keypoints,
                    'rightFrame': rightFrame,
                    'right_good_matches': right_good_matches,
                    'right_keypoints': right_keypoints,
                }

                if DEBUG:
                    left_output_frame = leftFeatureTracker.drawFeatures(leftFrame)
                    right_output_frame = rightFeatureTracker.drawFeatures(rightFrame)
                #     cv2.imshow("Left", left_output_frame)
                #     cv2.imshow("Right", right_output_frame)

            yield frame, bboxes, featuredata


def get_pipeline():
    return pipeline


def del_pipeline():
    del pipeline
