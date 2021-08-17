#!/usr/bin/env python3
import traceback
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from scipy import linalg
from collections import deque

from common.camera_info import CAMERA_LEFT, CAMERA_RIGHT
from common.config import NNConfig
from common.field_constants import LANDMARKS
from common.image_processing import output_perspective_transform

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(index_params, search_params)


def createRefImg():
    global src_kp
    global src_des
    global src_img
    maxCorners = max(25, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04

    src_img = cv2.imread('../resources/images/red_upper_power_port_sandbox.jpg')
    copy = np.copy(src_img)
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(copy, maxCorners, qualityLevel, minDistance, None,
                                      blockSize=blockSize, gradientSize=gradientSize,
                                      useHarrisDetector=useHarrisDetector, k=k)

    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]
    sift = cv2.SIFT_create()
    src_kp, src_des = sift.compute(src_img, keypoints)

    radius = 4
    for i in range(corners.shape[0]):
        cv2.circle(src_img, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius, (0, 0, 255), cv2.FILLED)


def solvePNPstereo(frame, camera_coefficients, matches, keypoints):
    global src_kp
    global src_img
    try:
        src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()
        corner_camera_coord, object_points_3d, center_pts = output_perspective_transform(src_img.shape, M)

        corner_camera_coord = corner_camera_coord.reshape(-1, 2)
        # solve pnp using iterative LMA algorithm
        retval, rVec, tVec, _ = cv2.solvePnPRansac(object_points_3d, corner_camera_coord,
                                                              camera_coefficients['intrinsicMatrix'],
                                                              camera_coefficients['distortionCoeff'])

        rotM = cv2.Rodrigues(rVec)[0]
        # translation = -np.matrix(rotM).T * np.matrix(tMatrix)
        # ppm = LANDMARKS['red_upper_power_port_sandbox']['width'] / (100)

        # translation = ppm * translation
        # rotation_deg = 57.2958 * rotation_rad

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        return tVec, rotM, draw_params, M

    except Exception as e:
        print("Error: {}".format(e))
        return [], [], [], []

# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
def DLT(cam1_matrix, cam2_matrix, point1, point2):
    A = [point1[1] * cam1_matrix[2, :] - cam1_matrix[1, :],
         cam1_matrix[0, :] - point1[0] * cam1_matrix[2, :],
         point2[1] * point2[2, :] - cam2_matrix[1, :],
         cam2_matrix[0, :] - point2[0] * cam2_matrix[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    print('Triangulated point: ')
    print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]

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

    def trackFeaturePath(self, features):

        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.position)
            while(len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)

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

    def matchRefImg(self, frame):
        global src_des
        keypoints = []
        for featurePath in self.trackedFeaturesPath.values():
            n = len(featurePath) - 1
            keypoints.append(cv2.KeyPoint(x=featurePath[n].x, y=featurePath[n].y, _size=20))

        target_keypoints, descriptors = sift.compute(frame, keypoints)

        if len(keypoints) < 2:
            return [], []

        matches = flann.knnMatch(src_des, descriptors, k=2)

        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        return good_matches, target_keypoints

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
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
featureTrackerLeft = pipeline.createFeatureTracker()
featureTrackerRight = pipeline.createFeatureTracker()

# rgbCamera = pipeline.createColorCamera()
# detectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
# objectTracker = pipeline.createObjectTracker()
# stereo = pipeline.createStereoDepth()

xoutPassthroughFrameLeft = pipeline.createXLinkOut()
xoutTrackedFeaturesLeft = pipeline.createXLinkOut()
xoutPassthroughFrameRight = pipeline.createXLinkOut()
xoutTrackedFeaturesRight = pipeline.createXLinkOut()
xinTrackedFeaturesConfig = pipeline.createXLinkIn()

# xlinkOut = pipeline.createXLinkOut()
# trackerOut = pipeline.createXLinkOut()

xoutPassthroughFrameLeft.setStreamName("passthroughFrameLeft")
xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
xoutPassthroughFrameRight.setStreamName("passthroughFrameRight")
xoutTrackedFeaturesRight.setStreamName("trackedFeaturesRight")
xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# xlinkOut.setStreamName("detections")
# trackerOut.setStreamName("tracklets")

# rgbCamera.setPreviewSize(416, 416)
# rgbCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# rgbCamera.setInterleaved(False)
# rgbCamera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# rgbCamera.setFps(30)

# stereo.setConfidenceThreshold(255)
# stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

# model_name = "infiniteRecharge2020sandbox"
# model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
# blob_path = model_dir / Path(model_name).with_suffix(f".blob")

# config_path = model_dir / Path(model_name).with_suffix(f".json")

# nn_config = NNConfig(config_path)
# labels = nn_config.labels

# detectionNetwork.setBlobPath(str(blob_path))
# detectionNetwork.setConfidenceThreshold(nn_config.confidence)
# detectionNetwork.setNumClasses(nn_config.metadata["classes"])
# detectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
# detectionNetwork.setAnchors(nn_config.metadata["anchors"])
# detectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
# detectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
# detectionNetwork.input.setBlocking(False)
# detectionNetwork.setBoundingBoxScaleFactor(0.5)
# detectionNetwork.setDepthLowerThreshold(100)
# detectionNetwork.setDepthUpperThreshold(50000)

# objectTracker.setDetectionLabelsToTrack([5])
# objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Linking
monoLeft.out.link(featureTrackerLeft.inputImage)
featureTrackerLeft.passthroughInputImage.link(xoutPassthroughFrameLeft.input)
featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)
xinTrackedFeaturesConfig.out.link(featureTrackerLeft.inputConfig)

monoRight.out.link(featureTrackerRight.inputImage)
featureTrackerRight.passthroughInputImage.link(xoutPassthroughFrameRight.input)
featureTrackerRight.outputFeatures.link(xoutTrackedFeaturesRight.input)
xinTrackedFeaturesConfig.out.link(featureTrackerRight.inputConfig)

# rgbCamera.preview.link(detectionNetwork.input)
# objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

# rgbCamera.video.link(objectTracker.inputTrackerFrame)
# detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

# detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
# detectionNetwork.out.link(objectTracker.inputDetections)
# objectTracker.out.link(trackerOut.input)

# monoLeft.out.link(stereo.left)
# monoRight.out.link(stereo.right)

# detectionNetwork.out.link(xlinkOut.input)
# stereo.depth.link(detectionNetwork.inputDepth)

# By default the least mount of resources are allocated
# increasing it improves performance
# numShaves = 2
# numMemorySlices = 2
# featureTrackerLeft.setHardwareResources(numShaves, numMemorySlices)
# featureTrackerRight.setHardwareResources(numShaves, numMemorySlices)

featureTrackerConfig = featureTrackerRight.initialConfig.get()
print("Press 's' to switch between Lucas-Kanade optical flow and hardware accelerated motion estimation!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues used to receive the results
    passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 8, False)
    outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)
    passthroughImageRightQueue = device.getOutputQueue("passthroughFrameRight", 8, False)
    outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 8, False)

    # preview = device.getOutputQueue("preview", 4, False)
    # tracklets = device.getOutputQueue("tracklets", 4, False)
    # detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    leftWindowName = "left"
    leftFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", leftWindowName)

    rightWindowName = "right"
    rightFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", rightWindowName)

    createRefImg()

    while True:
        # imgFrame = preview.get()
        # track = tracklets.get()
        # frame = imgFrame.getCvFrame()
        #
        # color = (255, 0, 0)
        # trackletsData = track.tracklets
        # for t in trackletsData:
        #     roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
        #     x1 = int(roi.topLeft().x)
        #     y1 = int(roi.topLeft().y)
        #     x2 = int(roi.bottomRight().x)
        #     y2 = int(roi.bottomRight().y)
        #
        #     cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #     cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        #
        # cv2.imshow("tracker", frame)

        inPassthroughFrameLeft = passthroughImageLeftQueue.get()
        passthroughFrameLeft = inPassthroughFrameLeft.getFrame()
        leftFrame = cv2.cvtColor(passthroughFrameLeft, cv2.COLOR_GRAY2BGR)

        inPassthroughFrameRight = passthroughImageRightQueue.get()
        passthroughFrameRight = inPassthroughFrameRight.getFrame()
        rightFrame = cv2.cvtColor(passthroughFrameRight, cv2.COLOR_GRAY2BGR)

        trackedFeaturesLeft = outputFeaturesLeftQueue.get().trackedFeatures
        leftFeatureDrawer.trackFeaturePath(trackedFeaturesLeft)
        leftFeatureDrawer.drawFeatures(leftFrame)
        left_good_matches, left_keypoints = leftFeatureDrawer.matchRefImg(leftFrame)

        trackedFeaturesRight = outputFeaturesRightQueue.get().trackedFeatures
        rightFeatureDrawer.trackFeaturePath(trackedFeaturesRight)
        rightFeatureDrawer.drawFeatures(rightFrame)
        right_good_matches, right_keypoints = rightFeatureDrawer.matchRefImg(rightFrame)


        # Show the frame
        cv2.imshow(leftWindowName, leftFrame)
        cv2.imshow(rightWindowName, rightFrame)

        left_translation = []
        right_translation = []
        if len(left_good_matches) > 5:
            left_translation, left_rotation, left_draw_params, left_M = solvePNPstereo(leftFrame, CAMERA_LEFT, left_good_matches, left_keypoints)

        if len(right_good_matches) > 5:
            right_translation, right_rotation, right_draw_params, right_M = solvePNPstereo(rightFrame, CAMERA_RIGHT, right_good_matches, right_keypoints)

        # Merge the two results with fancy math: https://stackoverflow.com/questions/51914161/solvepnp-vs-recoverpose-by-rotation-composition-why-translations-are-not-same
        rotation = [0, 0, 0]
        translation = [0, 0, 0]
        if any(left_translation) and any(right_translation):
            rotation = np.linalg.inv(left_rotation) * right_rotation
            translation = right_rotation * right_translation - left_rotation * left_translation
        elif any(left_translation):
            rotation = left_translation
            translation = left_translation
        elif any(right_translation):
            rotation = right_rotation
            translation = left_translation

        global src_img
        global src_kp
        try:
            img1 = np.copy(src_img)

            cv2.putText(img1, "x: {}".format(right_translation[0].round(2)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (255, 255, 255))
            cv2.putText(img1, "y: {}".format(right_translation[1].round(2)), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (255, 255, 255))
            # cv2.putText(img1, "z: {}".format(round(result['depth_z'], 2)), (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
            #             (255, 255, 255))
            cv2.putText(img1, "r: {}".format(right_rotation[0].round(2)), (0, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (255, 255, 255))
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if left_draw_params:
                dst = cv2.perspectiveTransform(pts, left_M)
                img2 = cv2.polylines(leftFrame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                output_frame = cv2.drawMatches(img1, src_kp, img2, left_keypoints, left_good_matches, None, **left_draw_params)
            elif right_draw_params:
                dst = cv2.perspectiveTransform(pts, right_M)
                img2 = cv2.polylines(rightFrame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                output_frame = cv2.drawMatches(img1, src_kp, img2, right_keypoints, right_good_matches, None, **right_draw_params)

            cv2.imshow("Ransac", output_frame)

        except Exception as e:
            print("Exception: {}".format(e))

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
