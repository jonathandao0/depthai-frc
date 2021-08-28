import json
import numpy as np


OAK_L_CALIBRATION_JSON = open('resources/14442C10218CCCD200.json')
OAK_L_CALIBRATION_DATA = json.load(OAK_L_CALIBRATION_JSON)

OAK_L_CAMERA_RGB = OAK_L_CALIBRATION_DATA['cameraData'][2][1]

OAK_L_CAMERA_LEFT = OAK_L_CALIBRATION_DATA['cameraData'][0][1]

OAK_L_CAMERA_RIGHT = OAK_L_CALIBRATION_DATA['cameraData'][1][1]
