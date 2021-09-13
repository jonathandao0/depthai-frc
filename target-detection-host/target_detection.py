import numpy as np


def find_target_center(edgeFrame, bbox):
    targetArea = edgeFrame[bbox['x_min']:bbox['x_max'], bbox['y_min']:bbox['y_max']]
    indicies = np.where(targetArea > 250)
    # values = edgeFrame[indicies]

    if len(indicies[0]) > 0:
        min_x = min(indicies[0])
        max_x = max(indicies[0])
    else:
        min_x = 0
        max_x = 0

    return ((max_x - min_x) / 2) + bbox['x_min']

# def parse_goal_frame(frame, bbox):
#     for bbox in bboxes:
#         if bbox['label']
#
#     return results
#
#
# def stream_frame():
#     pass
