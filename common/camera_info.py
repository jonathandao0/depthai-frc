import json

import cv2
import numpy as np


OAK_L_CALIBRATION_JSON = open('../resources/14442C10218CCCD200.json')
OAK_L_CALIBRATION_DATA = json.load(OAK_L_CALIBRATION_JSON)

OAK_L_CAMERA_RGB = OAK_L_CALIBRATION_DATA['cameraData'][2][1]

OAK_L_CAMERA_LEFT = OAK_L_CALIBRATION_DATA['cameraData'][0][1]

OAK_L_CAMERA_RIGHT = OAK_L_CALIBRATION_DATA['cameraData'][1][1]

LR_Translation = np.array(list(OAK_L_CAMERA_LEFT['extrinsics']['translation'].values())) / 100 # Convert cm to m
LR_Rotation = np.array(list(OAK_L_CAMERA_LEFT['extrinsics']['rotationMatrix']))

L_DISTORTION = np.array(OAK_L_CAMERA_LEFT['distortionCoeff'])
L_INTRINSIC = np.array(OAK_L_CAMERA_LEFT['intrinsicMatrix'])

R_DISTORTION = np.array(OAK_L_CAMERA_RIGHT['distortionCoeff'])
R_INTRINSIC = np.array(OAK_L_CAMERA_RIGHT['intrinsicMatrix'])

R1, R2, L_Projection, R_Projection, Q, L_ROI, R_ROI = cv2.stereoRectify(L_INTRINSIC, L_DISTORTION, R_INTRINSIC, R_DISTORTION, (1280, 720), LR_Rotation, LR_Translation)

OAK_L_PARAMS = {
    'l_intrinsic': L_INTRINSIC,
    'r_intrinsic': R_INTRINSIC,
    'l_distortion': L_DISTORTION,
    'r_distortion': R_DISTORTION,
    'l_projection': L_Projection,
    'r_projection': R_Projection
}
