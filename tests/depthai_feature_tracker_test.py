#!/usr/bin/env python3
import uuid
from pathlib import Path

import cv2
import depthai as dai
from collections import deque

from common.config import NNConfig


class FeatureTrackerDrawer:

    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    # for how many frames the feature is tracked
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    def onTrackBar(self, val):
        FeatureTrackerDrawer.trackedFeaturesPathLength = val
        pass

    def trackFeaturePath(self, features, bboxes):

        newTrackedIDs = set()
        featuresToRemove = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            exit = False
            if bboxes:
                for detection in bboxes:
                    y_min = int(detection['y_min'] * (2.0 / 3.0))
                    y_max = int(detection['y_max'] * (2.0 / 3.0))
                    x_min = int(detection['x_min'] * (2.0 / 3.0))
                    x_max = int(detection['x_max'] * (2.0 / 3.0))
                    point = currentFeature.position

                    if point.x < x_min or point.x > x_max or point.y < y_min or point.y > y_max:
                        featuresToRemove.add(currentID)

            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.position)
            while(len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            try:
                self.trackedFeaturesPath.pop(id)
            except Exception:
                pass

        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img):

        cv2.setTrackbarPos(self.trackbarName, self.windowName, FeatureTrackerDrawer.trackedFeaturesPathLength)

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath

            for j in range(len(path) - 1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j + 1].x), int(path[j + 1].y))
                cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA, 0)
            j = len(path) - 1
            cv2.circle(img, (int(path[j].x), int(path[j].y)), self.circleRadius, self.pointColor, -1, cv2.LINE_AA, 0)

    def __init__(self, trackbarName, windowName):
        self.trackbarName = trackbarName
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar(trackbarName, windowName, FeatureTrackerDrawer.trackedFeaturesPathLength, FeatureTrackerDrawer.maxTrackedFeaturesPathLength, self.onTrackBar)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
# rgbCenter = pipeline.createColorCamera()
featureTrackerLeft = pipeline.create(dai.node.FeatureTracker)
featureTrackerRight = pipeline.create(dai.node.FeatureTracker)
# featureTrackerCenter = pipeline.create(dai.node.FeatureTracker)

camRgb = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
stereo = pipeline.createStereoDepth()

xoutPassthroughFrameLeft = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesLeft = pipeline.create(dai.node.XLinkOut)
xoutPassthroughFrameRight = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesRight = pipeline.create(dai.node.XLinkOut)
# xoutPassthroughFrameCenter = pipeline.create(dai.node.XLinkOut)
# xoutTrackedFeaturesCenter = pipeline.create(dai.node.XLinkOut)
xinTrackedFeaturesConfig = pipeline.create(dai.node.XLinkIn)

xoutRgb = pipeline.createXLinkOut()
camRgb.preview.link(xoutRgb.input)
xoutNN = pipeline.createXLinkOut()

xoutPassthroughFrameLeft.setStreamName("passthroughFrameLeft")
xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
xoutPassthroughFrameRight.setStreamName("passthroughFrameRight")
xoutTrackedFeaturesRight.setStreamName("trackedFeaturesRight")
# xoutPassthroughFrameCenter.setStreamName("passthroughFrameCenter")
# xoutTrackedFeaturesCenter.setStreamName("trackedFeaturesCenter")
xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# rgbCenter.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# rgbCenter.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

stereo.setConfidenceThreshold(255)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

model_name = "infiniteRecharge2020sandbox"
model_dir = Path(__file__).parent / Path(f"../resources/nn/") / model_name
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
spatialDetectionNetwork.setDepthUpperThreshold(5000)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")

# Linking
monoLeft.out.link(featureTrackerLeft.inputImage)
featureTrackerLeft.passthroughInputImage.link(xoutPassthroughFrameLeft.input)
featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)
xinTrackedFeaturesConfig.out.link(featureTrackerLeft.inputConfig)

monoRight.out.link(featureTrackerRight.inputImage)
featureTrackerRight.passthroughInputImage.link(xoutPassthroughFrameRight.input)
featureTrackerRight.outputFeatures.link(xoutTrackedFeaturesRight.input)
xinTrackedFeaturesConfig.out.link(featureTrackerRight.inputConfig)

# rgbCenter.video.link(featureTrackerCenter.inputImage)
# featureTrackerCenter.passthroughInputImage.link(xoutPassthroughFrameCenter.input)
# featureTrackerCenter.outputFeatures.link(xoutTrackedFeaturesCenter.input)
# xinTrackedFeaturesConfig.out.link(featureTrackerCenter.inputConfig)

# By default the least mount of resources are allocated
# increasing it improves performance

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)

spatialDetectionNetwork.out.link(xoutNN.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

numShaves = 2
numMemorySlices = 2
featureTrackerLeft.setHardwareResources(numShaves, numMemorySlices)
featureTrackerRight.setHardwareResources(numShaves, numMemorySlices)
# featureTrackerCenter.setHardwareResources(numShaves, numMemorySlices)

featureTrackerConfig = featureTrackerRight.initialConfig.get()
print("Press 's' to switch between Lucas-Kanade optical flow and hardware accelerated motion estimation!")

detections = []

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues used to receive the results
    passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 8, False)
    outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)
    passthroughImageRightQueue = device.getOutputQueue("passthroughFrameRight", 8, False)
    outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 8, False)
    # passthroughImageCenterQueue = device.getOutputQueue("passthroughFrameCenter", 8, False)
    # outputFeaturesCenterQueue = device.getOutputQueue("trackedFeaturesCenter", 8, False)
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    leftWindowName = "left"
    leftFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", leftWindowName)

    rightWindowName = "right"
    rightFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", rightWindowName)

    # centerWindowName = "center"
    # centerFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", centerWindowName)

    while True:
        frame = previewQueue.get().getCvFrame()
        inDet = detectionNNQueue.tryGet()

        if inDet is not None:
            detections = inDet.detections

        bboxes = []
        height = frame.shape[0]
        width = frame.shape[1]
        for detection in detections:
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

        for detection in bboxes:
            cv2.rectangle(frame, (detection['x_min'], detection['y_min']), (detection['x_max'], detection['y_max']), (0, 255, 0), 2)
            cv2.putText(frame, "x: {}".format(round(detection['depth_x'], 2)), (detection['x_min'], detection['y_min'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "y: {}".format(round(detection['depth_y'], 2)), (detection['x_min'], detection['y_min'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "z: {}".format(round(detection['depth_z'], 2)), (detection['x_min'], detection['y_min'] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "conf: {}".format(round(detection['confidence'], 2)), (detection['x_min'], detection['y_min'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "label: {}".format(LABELS[detection['label']], 1), (detection['x_min'], detection['y_min'] + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            cv2.imshow("Frame", frame)

        inPassthroughFrameLeft = passthroughImageLeftQueue.get()
        passthroughFrameLeft = inPassthroughFrameLeft.getFrame()
        leftFrame = cv2.cvtColor(passthroughFrameLeft, cv2.COLOR_GRAY2BGR)
        if bboxes:
            for detection in bboxes:
                y_min = int(detection['y_min'] * (2.0 / 3.0))
                y_max = int(detection['y_max'] * (2.0 / 3.0))
                x_min = int(detection['x_min'] * (2.0 / 3.0))
                x_max = int(detection['x_max'] * (2.0 / 3.0))
                # leftFrame = leftFrame[y_min:y_max, x_min:x_max]

        inPassthroughFrameRight = passthroughImageRightQueue.get()
        passthroughFrameRight = inPassthroughFrameRight.getFrame()
        rightFrame = cv2.cvtColor(passthroughFrameRight, cv2.COLOR_GRAY2BGR)

        # inPassthroughFrameCenter = passthroughImageCenterQueue.get()
        # passthroughFrameCenter = inPassthroughFrameCenter.getFrame()
        # centerFrame = cv2.cvtColor(passthroughFrameCenter, cv2.COLOR_GRAY2BGR)

        trackedFeaturesLeft = outputFeaturesLeftQueue.get().trackedFeatures
        leftFeatureDrawer.trackFeaturePath(trackedFeaturesLeft, bboxes)
        leftFeatureDrawer.drawFeatures(leftFrame)

        trackedFeaturesRight = outputFeaturesRightQueue.get().trackedFeatures
        rightFeatureDrawer.trackFeaturePath(trackedFeaturesRight, bboxes)
        rightFeatureDrawer.drawFeatures(rightFrame)

        # trackedFeaturesCenter = outputFeaturesCenterQueue.get().trackedFeatures
        # centerFeatureDrawer.trackFeaturePath(trackedFeaturesCenter)
        # centerFeatureDrawer.drawFeatures(centerFrame)

        # Show the frame
        cv2.imshow(leftWindowName, leftFrame)
        cv2.imshow(rightWindowName, rightFrame)
        # cv2.imshow(centerWindowName, centerFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if featureTrackerConfig.motionEstimator.type == dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.HW_MOTION_ESTIMATION
                print("Switching to hardware accelerated motion estimation")
            else:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW
                print("Switching to Lucas-Kanade optical flow")

            cfg = dai.FeatureTrackerConfig()
            cfg.set(featureTrackerConfig)
            inputFeatureTrackerConfigQueue.send(cfg)