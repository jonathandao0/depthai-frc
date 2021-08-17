
import argparse
import cv2
import logging
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='detector', help='Set detector to use', default='SIFT', type=str.upper)
parser.add_argument('-i', dest='img', help='Set image to use', default='red_upper_power_port_sandbox.jpg', type=str.lower)
args = parser.parse_args()

log = logging.getLogger(__name__)

default_img_path = '../resources/images/red_upper_power_port_sandbox.jpg'


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

def main():
    tmp_path = '../resources/images/' + args.img
    if os.path.exists(tmp_path):
        img_path = tmp_path
    else:
        img_path = default_img_path

    if args.detector == 'SURF':
        surf(img_path)
    elif args.detector == 'FAST':
        fast(img_path)
    elif args.detector == 'BRIEF':
        brief(img_path)
    elif args.detector == 'ORB':
        orb(img_path)
    else:
        sift(img_path)

    while True:
        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    main()
