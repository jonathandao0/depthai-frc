import logging
import math

import cv2
import numpy as np

from common.camera_info import OAK_L_CAMERA_RIGHT, OAK_L_CAMERA_LEFT
from common.field_constants import LANDMARKS

sift = cv2.SIFT_create()
SIFT_PARAMS = {}

log = logging.getLogger(__name__)


def initalizeAllRefTargets():
    landmarks = ['red_upper_power_port', 'red_loading_bay', 'blue_upper_power_port', 'blue_loading_bay']
    for landmark in landmarks:
        try:
            img_path = "../resources/images/{}.jpg".format(landmark)
            image, keypoints, descriptors = createRefImg(img_path)
            # points3D = createPoints3D(landmark, params, keypoints)
            SIFT_PARAMS[landmark] = {
                'image': image,
                'keypoints': keypoints,
                # '3D_points': points3D,
                'descriptors': descriptors
            }
        except Exception as e:
            log.error("Couldn't generate sift params for {}: {}".format(landmark, e))
            pass


def createRefImg(img_path):
    maxCorners = max(25, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04

    src_img = cv2.imread(img_path)
    copy = np.copy(src_img)
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(copy, maxCorners, qualityLevel, minDistance, None,
                                      blockSize=blockSize, gradientSize=gradientSize,
                                      useHarrisDetector=useHarrisDetector, k=k)

    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in corners]
    img_kp, img_des = sift.compute(src_img, keypoints)

    radius = 4
    for i in range(corners.shape[0]):
        cv2.circle(src_img, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius, (0, 0, 255), cv2.FILLED)

    return src_img, img_kp, img_des

# def createPoints3D(landmark, params, keypoints):
#     ppm =
#
#     return points3D

def drawSolvePNP(src_frame, dst_frame, src_kp, dst_kp, dst_good_matches, M, draw_params):
    h, w, d = src_frame.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(dst_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    output_frame = cv2.drawMatches(src_frame, src_kp, img2, dst_kp, dst_good_matches, None, **draw_params)

    return output_frame


def solvePNPStereo(src_img, src_kp, good_matches_threshold, left_keypoints, left_good_matches, right_keypoints, right_good_matches):

    # Triangulate points

    # Use triangulated 3D points with one of the stereo 2D points for solvePNP
    retval, rVec, tVec, _ = cv2.solvePnPRansac(triangulated_points_3d, left_keypoints,
                                               OAK_L_CAMERA_LEFT['intrinsicMatrix'],
                                               OAK_L_CAMERA_LEFT['distortionCoeff'])

    rotM = cv2.Rodrigues(rVec)[0]

    return retval, tVec, rotM
    # return [], [], []


# https://github.com/GigaFlopsis/image_pose_estimation
def output_perspective_transform(img_shape, M):
    h, w = img_shape[:2]
    corner_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    center_pts = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    corner_pts_3d = np.float32(
        [[-w / 2, -h / 2, 0], [-w / 2, (h - 1) / 2, 0], [(w - 1) / 2, (h - 1) / 2, 0], [(w - 1) / 2, -h / 2, 0]])  ###
    corner_camera_coord = cv2.perspectiveTransform(corner_pts, M)  ###
    # center_camera_coord = cv2.perspectiveTransform(center_pts, M)

    return corner_camera_coord, corner_pts_3d, center_pts


# https://answers.opencv.org/question/161369/retrieve-yaw-pitch-roll-from-rvec/
def decomposeYawPitchRoll(R):
    sin_x = math.sqrt(R[2, 0] * R[2, 0] + R[2, 1] * R[2, 1])
    validity = sin_x < 1e-6
    if not validity:
        z1 = math.atan2(R[2, 0], R[2, 1])  # around z1-axis
        x = math.atan2(sin_x, R[2, 2])  # around x-axis
        z2 = math.atan2(R[0, 2], -R[1, 2])  # around z2-axis
    else:  # gimbal lock
        z1 = 0  # around z1-axis
        x = math.atan2(sin_x, R[2, 2])  # around x-axis
        z2 = 0  # around z2-axis

    return np.array([[z1], [x], [z2]])


def maskImageThreshold(frame, threshold):
    mask = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)[1]

    return cv2.bitwise_and(frame, mask)