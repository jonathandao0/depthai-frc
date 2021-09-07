import uuid
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from common.config import *
from imutils.video import FPS

from common.feature_tracker import FeatureTrackerDebug, FeatureTracker, matchStereoToRefImage, calculateRotationMask
from common.image_processing import SIFT_PARAMS, drawSolvePNP

log = logging.getLogger(__name__)

LABELS = []


def create_pipeline(model_name):
    global pipeline
    global LABELS
    global disparityMultiplier
    log.info("Creating DepthAI pipeline...")

    # out_depth = False  # Disparity by default
    # out_rectified = True  # Output and display rectified streams
    # lrcheck = True  # Better handling for occlusions
    # extended = False  # Closer-in minimum depth, disparity range is doubled
    # subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
    # # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

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
    disparityOut = pipeline.createXLinkOut()
    depthOut = pipeline.createXLinkOut()

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
    disparityOut.setStreamName('disparity')
    depthOut.setStreamName('depth')

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Setting node configs
    # stereo.setOutputDepth(out_depth)
    # stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(255)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.setMedianFilter(median) # KERNEL_7x7 default
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
    stereo.disparity.link(disparityOut.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(depthOut.input)

    disparityMultiplier = 255 / stereo.getMaxDisparity()
    log.info("Pipeline created.")

    return pipeline, LABELS


def capture():
    with dai.Device(pipeline) as device:
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

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
            results = {}
            frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()
            inDisparity = disparityQueue.tryGet()
            inDepth = depthQueue.tryGet()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            disparityFrame = np.array([])
            if inDisparity is not None:
                # Flip disparity frame, normalize it and apply color map for better visualization
                disparityFrame = inDisparity.getCvFrame()
                disparityFrame = cv2.flip(disparityFrame, 1)
                disparityFrame = (disparityFrame * disparityMultiplier).astype(np.uint8)
                disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)

            depth_map = np.array([])
            if inDepth is not None:
                depth_map = inDepth.getCvFrame().astype(np.uint16)

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

                trackedFeaturesRight = outputFeaturesRightQueue.get().trackedFeatures
                rightFeatureTracker.trackFeaturePath(trackedFeaturesRight)

                featuredata = {}
                if any(trackedFeaturesLeft) and any(trackedFeaturesRight):
                    ref_good_matches, left_keypoints, left_filtered_keypoints, right_filtered_keypoints = matchStereoToRefImage(trackedFeaturesLeft, leftFrame, trackedFeaturesRight, rightFrame, target_sift_params["descriptors"])

                    featuredata[data['id']] = {
                        'frame': leftFrame,
                        'good_matches': ref_good_matches,
                        'left_keypoints': left_filtered_keypoints,
                        'right_keypoints': right_filtered_keypoints
                    }

                    results['featuredata'] = featuredata

                    if DEBUG:
                        if any(ref_good_matches):
                            src_pts = np.float32([target_sift_params['keypoints'][m.queryIdx].pt for m in ref_good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([left_keypoints[m.trainIdx].pt for m in ref_good_matches]).reshape(-1, 1, 2)

                            if len(src_pts) >= 4 and len(dst_pts) >=4:
                                try:
                                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                                    matchesMask = mask.ravel().tolist()

                                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                                       singlePointColor=None,
                                                       matchesMask=matchesMask,  # draw only inliers
                                                       flags=2)

                                    h, w, d = target_sift_params['image'].shape
                                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                                    dst = cv2.perspectiveTransform(pts, M)
                                    img2 = cv2.polylines(leftFrame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                                    img3 = cv2.drawMatches(target_sift_params['image'], target_sift_params['keypoints'], img2, left_keypoints,
                                                           ref_good_matches, None, **draw_params)

                                    cv2.imshow("Ransac", img3)
                                except Exception as e:
                                    pass

            results['frame'] = frame
            results['bboxes'] = bboxes
            results['disparityFrame'] = disparityFrame

            yield results


def get_pipeline():
    return pipeline


def del_pipeline():
    del pipeline
