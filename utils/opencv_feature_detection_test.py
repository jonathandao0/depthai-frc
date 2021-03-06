
import argparse
import cv2
import logging
import numpy as np
import os

default_img_name = 'red_upper_power_port_sandbox'
# default_img_name = 'official_blue_lower_power_port'

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='detector', help='Set detector to use', default='SIFT_CUSTOM', type=str.upper)
parser.add_argument('-i', dest='img', help='Set image to use', default=default_img_name, type=str.lower)
args = parser.parse_args()

log = logging.getLogger(__name__)


def sift(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    cv2.imshow("SIFT", img)


# Doesn't work, requires payment
def surf(img_path):
    img = cv2.imread(img_path, 0)

    surf = cv2.xfeatures2d.SURF_create(400)

    kp, des = surf.detectAndCompute(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    cv2.imshow("SURF", img2)


def fast(img_path):
    fast = cv2.FastFeatureDetector_create()
    img = cv2.imread(img_path, 0)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)

    img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv2.imshow("FAST - 1", img2)
    cv2.imshow("FAST - 2", img3)


# Doesn't work, requires payment
def brief(img_path):
    img = cv2.imread(img_path, 0)
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img, None)
    kp, des = brief.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)

    cv2.imshow("BRIEF", img2)


def orb(img_path):
    img = cv2.imread(img_path, 0)
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

    cv2.imshow("ORB", img2)


def sift_custom(img_path):
    maxCorners = max(25, 1)
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
    img_kp = img_kp[0:1]
    img2 = cv2.drawKeypoints(img, img_kp, None, color=(0, 255, 0), flags=0)

    h, w, d = img.size
    ppm = / w
    points3D =

    cv2.imshow("SIFT_CUSTOM", img2)
    pass


def main():
    tmp_path = '../resources/images/' + args.img
    if os.path.exists(tmp_path):
        img_path = tmp_path
    else:
        img_path = '../resources/images/' + default_img_name + '.jpg'

    if args.detector == 'SIFT':
        sift(img_path)
    elif args.detector == 'SURF':
        surf(img_path)
    elif args.detector == 'FAST':
        fast(img_path)
    elif args.detector == 'BRIEF':
        brief(img_path)
    elif args.detector == 'ORB':
        orb(img_path)
    else:
        sift_custom(img_path)

    while True:
        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    main()
