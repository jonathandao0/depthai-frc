from collections import deque

import cv2

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

    def matchRefImg(self, frame, src_des):
        good_matches = []
        keypoints = []

        for featurePath in self.trackedFeaturesPath.values():
            n = len(featurePath) - 1
            keypoints.append(cv2.KeyPoint(x=featurePath[n].x, y=featurePath[n].y, _size=20))

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
