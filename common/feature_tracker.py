from collections import deque

import cv2

import numpy as np

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(index_params, search_params)


class FeatureTracker:

    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    # for how many frames the feature is tracked
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    def trackFeaturePath(self, features):

        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.position)
            while(len(path) > max(1, FeatureTracker.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)

        self.trackedIDs = newTrackedIDs

    def matchRefImg(self, frame, keypoints, src_des):
        good_matches = []

        target_kp, target_des = sift.compute(frame, keypoints)

        if len(keypoints) >= 2:
            matches = flann.knnMatch(src_des, target_des, k=2)

            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches, target_kp

    def __init__(self):
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()


class FeatureTrackerDebug(FeatureTracker):

    def onTrackBar(self, val):
        FeatureTrackerDebug.trackedFeaturesPathLength = val
        pass

    def drawFeatures(self, img):

        cv2.setTrackbarPos(self.trackbarName, self.windowName, FeatureTrackerDebug.trackedFeaturesPathLength)

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath

            # for j in range(len(path) - 1):
            #     src = (int(path[j].x), int(path[j].y))
            #     dst = (int(path[j + 1].x), int(path[j + 1].y))
            #     cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA, 0)
            j = len(path) - 1
            cv2.circle(img, (int(path[j].x), int(path[j].y)), self.circleRadius, self.pointColor, -1, cv2.LINE_AA, 0)

        return  img

    def __init__(self, trackbarName, windowName):
        super().__init__()

        self.trackbarName = trackbarName
        self.windowName = windowName
        # cv2.namedWindow(windowName)
        # cv2.createTrackbar(trackbarName, windowName, FeatureTrackerDebug.trackedFeaturesPathLength, FeatureTrackerDebug.maxTrackedFeaturesPathLength, self.onTrackBar)


def matchStereoToRefImage(trackedFeaturesLeft, leftFrame, trackedFeaturesRight, rightFrame, refDes):
    left_keypoints = []
    right_keypoints = []
    ref_stereo_matches = []
    left_good_kp = []
    right_good_kp = []

    for feature in trackedFeaturesLeft:
        left_keypoints.append(cv2.KeyPoint(x=feature.position.x, y=feature.position.y, size=20))

    left_kp, left_des = sift.compute(leftFrame, left_keypoints)

    for feature in trackedFeaturesRight:
        right_keypoints.append(cv2.KeyPoint(x=feature.position.x, y=feature.position.y, size=20))

    right_kp, right_des = sift.compute(rightFrame, right_keypoints)

    if len(left_des) >= 2 and len(right_des) >= 2:
        left_right_matches = flann.knnMatch(left_des, right_des, k=2)

        left_right_kp1 = []
        left_right_des = []
        left_right_kp2 = []
        for m, n in left_right_matches:
            if m.distance < 0.7 * n.distance:
                left_right_kp1.append(left_kp[m.queryIdx])
                left_right_des.append(left_des[m.queryIdx])
                left_right_kp2.append(right_kp[m.trainIdx])

        left_right_des = np.array(left_right_des)

        if len(refDes) >= 2 and len(left_right_des) >= 2:
            stereo_matches = flann.knnMatch(refDes, left_right_des, k=2)

            for m, n in stereo_matches:
                if m.distance < 0.7 * n.distance:
                    ref_stereo_matches.append(m)
                    left_good_kp.append(left_right_kp1[m.trainIdx])
                    right_good_kp.append(left_right_kp2[m.trainIdx])

    return ref_stereo_matches, left_right_kp1, left_good_kp, right_good_kp


def calculateRotationMask(src_kp, dst_kp, matches):
    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    return M, draw_params