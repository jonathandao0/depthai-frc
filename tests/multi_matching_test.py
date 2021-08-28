import cv2
import numpy as np


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(index_params, search_params)

def sift_custom(img_path, name, max_points=25):
    maxCorners = max(max_points, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04

    img = cv2.imread(img_path)
    copy = np.copy(img)
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(copy, maxCorners, qualityLevel, minDistance, None,
                                      blockSize=blockSize, gradientSize=gradientSize,
                                      useHarrisDetector=useHarrisDetector, k=k)

    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]
    sift = cv2.SIFT_create()
    img_kp, img_des = sift.compute(img, keypoints)

    img2 = cv2.drawKeypoints(img, img_kp, None, color=(0, 255, 0), flags=0)

    # cv2.imshow(name, img2)

    return img_kp, img_des, img

def main():
    ref_img = '../resources/images/red_upper_power_port_sandbox.jpg'
    imgA = '../resources/images/red_upper_power_port_sandbox_A.jpg'
    imgB = '../resources/images/red_upper_power_port_sandbox_B.jpg'

    kp_R, des_R, img_R = sift_custom(ref_img, "REF")
    kp_A, des_A, img_A = sift_custom(imgA, "A", 125)
    kp_B, des_B, img_B = sift_custom(imgB, "B", 300)

    matches = flann.knnMatch(des_A, des_B, k=2)

    good = []
    kp_A_1 = []
    des_A_1 = []
    # kp_B_1 = []
    # des_B_1 = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
            kp_A_1.append(kp_A[m.queryIdx])
            des_A_1.append(des_A[m.queryIdx])
            # kp_B_1.append(kp_B[m.trainIdx])
            # des_B_1.append(des_B[m.queryIdx])

    des_A_1 = np.array(des_A_1)
    # des_B_1 = np.array(des_B_1)

    matches2 = flann.knnMatch(des_R, des_A_1, k=2)

    good2 = []
    for m, n in matches2:
        if m.distance < 0.8 * n.distance:
            good2.append([m])

    img5 = cv2.drawMatchesKnn(img_R, kp_R, img_A, kp_A_1, good2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("A-B-R matches", img5)

    img1 = cv2.drawKeypoints(img_R, kp_R, None, color=(0, 255, 0), flags=0)
    cv2.imshow("REF", img1)

    img2 = cv2.drawMatchesKnn(img_A, kp_A, img_B, kp_B, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("A-B matches", img2)

    # img3 = cv2.drawKeypoints(img_A, kp_A_1, None, color=(0, 255, 0), flags=0)
    # img4 = cv2.drawKeypoints(img_B, kp_B_1, None, color=(0, 255, 0), flags=0)

    # cv2.imshow("A1 matches", img3)
    # cv2.imshow("B1 matches", img4)

    while True:
        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()



main()
