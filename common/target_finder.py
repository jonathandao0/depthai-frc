#!/usr/bin/env python3
import math

import cv2
import numpy as np

from common.config import NN_IMG_SIZE


def find_largest_contour(edgeFrame, bbox):
    thresh = cv2.threshold(edgeFrame[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], 25, 255, cv2.THRESH_BINARY)[1]
    res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[-2]
    # contours = res[0] if len(res) == 2 else res[1]
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        result = np.zeros_like(edgeFrame)
        cv2.drawContours(result[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], [largest_contour], 0, (255, 255, 255), cv2.FILLED)

        rect = cv2.minAreaRect(largest_contour)
        center, _, _ = rect
        center_x, center_y = center

        edgeFrame = cv2.bitwise_or(edgeFrame, result)

        return edgeFrame, center_x + bbox['x_min'], center_y + bbox['y_min']
    else:
        return edgeFrame, -999, -999


def find_largest_hexagon_contour(edgeFrame, bbox):
    padding = 20
    y_min = 0 if bbox['y_min'] - padding < 0 else bbox['y_min']
    y_max = NN_IMG_SIZE if bbox['y_max'] + padding > NN_IMG_SIZE else bbox['y_max']
    x_min = 0 if bbox['x_min'] - padding < 0 else bbox['x_min']
    x_max = NN_IMG_SIZE if bbox['x_max'] + padding > NN_IMG_SIZE else bbox['x_max']
    area = (x_max - x_min) * (y_max - y_min)

    thresh = cv2.threshold(edgeFrame[y_min:y_max, x_min:x_max], 50, 255, cv2.THRESH_BINARY)[1]

    res = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = res[-2]
    contours = res[0] if len(res) == 2 else res[1]
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            poly = cv2.convexHull(c)
            peri = cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, 0.03 * peri, True)
            contourArea = cv2.contourArea(poly)

            if len(approx) == 6 and 0.9 > (contourArea / area) > 0.1:
                cv2.drawContours(edgeFrame[y_min:y_max, x_min:x_max], [c], 0, (255, 255, 255), cv2.FILLED)

                rect = cv2.minAreaRect(c)
                center, _, _ = rect
                center_x, center_y = center

                return edgeFrame, center_x + bbox['x_min'], center_y + bbox['y_min']

    return edgeFrame, -999, -999


def find_largest_circular_contour(edgeFrame, bbox):
    thresh = cv2.threshold(edgeFrame[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], 40, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("test", thresh)
    res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[-2]
    # contours = res[0] if len(res) == 2 else res[1]
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            max_r = min((bbox['y_max'] - bbox['y_min']) / 2, (bbox['x_max'] - bbox['x_min']) / 2)
            if len(approx) > 8 and r < max_r:
                cv2.circle(edgeFrame[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], (int(x), int(y)), int(r), (255, 255, 255), cv2.FILLED)
                # cv2.drawContours(edgeFrame[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], [contour], -1, (255, 255, 255), cv2.FILLED)

                rect = cv2.minAreaRect(contour)
                center, _, _ = rect
                center_x, center_y = center

                return edgeFrame, center_x + bbox['x_min'], center_y + bbox['y_min']

    return edgeFrame, -999, -999


def find_target_center(edgeFrame, bbox):
    # targetArea = edgeFrame[bbox['x_min']:bbox['x_max'], bbox['y_min']:bbox['y_max']]
    indicies = np.where(edgeFrame > 100)
    # values = edgeFrame[indicies]

    if len(indicies[0]) > 0:
        min_x = min(indicies[0])
        max_x = max(indicies[0])

        # x_offset = min_x + bbox['x_min']
        x_offset = min_x + bbox['x_min']
        return ((max_x - min_x) / 2) + x_offset, max_x + bbox['x_min'], min_x + bbox['x_min']
    else:
        return 9999, 0, 0


# def parse_goal_frame(frame, bbox):
#     for bbox in bboxes:
#         if bbox['label']
#
#     return results
#
#
# def stream_frame():
#     pass
