import cv2
import numpy as np


def homography_to_perspective_transform(img_shape, M):
    h, w = img_shape[:2]
    corner_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    center_pts = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    corner_pts_3d = np.float32(
        [[-w / 2, -h / 2, 0], [-w / 2, (h - 1) / 2, 0], [(w - 1) / 2, (h - 1) / 2, 0], [(w - 1) / 2, -h / 2, 0]])  ###
    corner_camera_coord = cv2.perspectiveTransform(corner_pts, M)  ###
    center_camera_coord = cv2.perspectiveTransform(center_pts, M)

    return corner_camera_coord, corner_pts_3d, center_pts