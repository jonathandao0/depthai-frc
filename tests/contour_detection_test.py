
import cv2
import math
import numpy as np


img = cv2.imread('../resources/images/goal_edge_test.PNG')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)[1]

res = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = res[0] if len(res) == 2 else res[1]
if len(contours) > 0:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = img_gray.shape
    area = h * w
    for i, c in enumerate(contours):
        c_area = cv2.contourArea(c)
        if area * 0.8 > cv2.contourArea(c) > area * 0.1:
            blank = np.zeros((h, w))
            cv2.drawContours(blank, [c], 0, (255, 255, 255), cv2.FILLED)
            cv2.imshow("C {}".format(i), blank)

cv2.imshow("gray", img_gray)

while True:
    key = cv2.waitKey(1)

    if key == ord("q"):
        raise StopIteration()