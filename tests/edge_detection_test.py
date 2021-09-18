import cv2
import numpy as np

bbox = {
    'x_min': 112,
    'x_max': 382,
    'y_min': 157,
    'y_max': 378,
}
edgeFrame = cv2.imread('../resources/images/goal_edge_detection_test.PNG', cv2.IMREAD_GRAYSCALE)

thresh = cv2.threshold(edgeFrame[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], 40, 255, cv2.THRESH_BINARY)[1]
res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = res[-2]
# contours = res[0] if len(res) == 2 else res[1]
if len(contours) > 0:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = max(contours, key=cv2.contourArea)

    result = np.zeros_like(edgeFrame)
    cv2.drawContours(result[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], [largest_contour], 0,
                     (255, 255, 255), cv2.FILLED)

    rect = cv2.minAreaRect(largest_contour)
    center, _, _ = rect
    center_x, center_y = center

    edgeFrame = cv2.bitwise_or(edgeFrame, result)

    edgeFrame = cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (150, 150, 150), 2)
    blankFrame = np.zeros_like(edgeFrame)
    contourFrame = cv2.drawContours(blankFrame, contours[0:2], -1, (255, 255, 255), 1)

while True:
    cv2.imshow("edges", edgeFrame)
    cv2.imshow("edgesR", result)

    cv2.imshow("largest", contourFrame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        raise StopIteration()
