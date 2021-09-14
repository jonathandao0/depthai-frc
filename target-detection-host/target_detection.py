import cv2
import numpy as np


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
        return edgeFrame, 0


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
