import math

import cv2
import numpy as np

from common.camera_info import CAMERA_LEFT, CAMERA_RIGHT
from common.field_constants import LANDMARKS

sift = cv2.SIFT_create()
SIFT_PARAMS = {}


def initalizeAllRefTargets():
    for landmark, params in LANDMARKS.items():
        try:
            img_path = "../resources/images/{}.jpg".format(landmark)
            image, keypoints, descriptors = createRefImg(img_path)
            SIFT_PARAMS[landmark] = {
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors
            }
        except Exception:
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

    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]
    img_kp, img_des = sift.compute(src_img, keypoints)

    radius = 4
    for i in range(corners.shape[0]):
        cv2.circle(src_img, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius, (0, 0, 255), cv2.FILLED)

    return src_img, img_kp, img_des


def drawSolvePNP(src_frame, dst_frame, src_kp, dst_kp, dst_good_matches, draw_params):
    h, w, d = src_frame.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, left_M)
    img2 = cv2.polylines(dst_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    output_frame = cv2.drawMatches(src_frame, src_kp, img2, dst_kp, dst_good_matches, None, **draw_params)

    return output_frame


def solvePNPStereo(src_img, src_kp, good_matches_threshold, left_keypoints, left_good_matches, right_keypoints, right_good_matches):
    retval_left = False
    left_translation = []
    left_draw_params = []
    left_M = []
    retval_right = False
    right_translation = []
    right_draw_params = []
    right_M = []

    if len(left_good_matches) > good_matches_threshold:
        retval_left, left_translation, left_rotation, left_draw_params, left_M = solvePNPkeypoints(src_img, src_kp, CAMERA_LEFT, left_good_matches, left_keypoints)

    if len(right_good_matches) > good_matches_threshold:
        retval_right, right_translation, right_rotation, right_draw_params, right_M = solvePNPkeypoints(src_img, src_kp, CAMERA_RIGHT, right_good_matches, right_keypoints)

    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    # Merge two results with fancy math: https://stackoverflow.com/questions/51914161/solvepnp-vs-recoverpose-by-rotation-composition-why-translations-are-not-same
    if retval_left and retval_right:
        rotation = np.linalg.inv(left_rotation) * right_rotation
        translation = np.dot(right_rotation, right_translation) - np.dot(left_rotation, left_translation)
    elif retval_left:
        rotation = left_rotation
        translation = left_translation
    elif retval_right:
        rotation = right_rotation
        translation = right_translation

    retval = retval_left or retval_right

    draw_params = {
        'left_draw_params': left_draw_params,
        'left_M': left_M,
        'right_draw_params': right_draw_params,
        'right_M': right_M,
    }

    return retval, translation, rotation, draw_params


def solvePNPkeypoints(src_img, src_kp, camera_coefficients, matches, keypoints):
    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    corner_camera_coord, object_points_3d, center_pts = output_perspective_transform(src_img.shape, M)

    corner_camera_coord = corner_camera_coord.reshape(-1, 2)

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

    return retval, tVec, rotM, draw_params, M

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