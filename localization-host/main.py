import argparse
import logging

import cv2
import depthai_utils
import numpy as np

from common import image_processing
from common.config import MODEL_NAME, DEBUG, NN_IMG_SIZE, DETECTION_PADDING
from common.field2dwindow import Field2dWindow
from distance import DistanceCalculations, DistanceCalculationsDebug
from common.field_constants import *

from common.networktables_client import NetworkTablesClient
from common.image_processing import output_perspective_transform, SIFT_PARAMS
from common.camera_info import CAMERA_RGB
from utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:
    distance_class = DistanceCalculations
    nt_class = NetworkTablesClient

    labels = None

    robot_pose3d = None
    has_targets = False

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __init__(self):
        self.nt_client = self.nt_class("localhost", "Depthai")
        self.pipeline, self.labels = depthai_utils.create_pipeline(MODEL_NAME)
        self.fps = FPSHandler()

        self.robot_pose3d = [FIELD_HEIGHT / 2, FIELD_WIDTH / 2, 0, 0]
        self.last_pose_update_time = 0

        image_processing.initalizeAllRefTargets()

    def parse_frame(self, frame, results, featuredata):
        self.fps.tick('frame')
        # distance_results = self.distance_class.parse_frame(frame, results)
        for result in results:
            self.has_targets = True

            try:
                label_name = depthai_utils.LABELS[result['label']]
                sift_params = SIFT_PARAMS[label_name]
                img1 = sift_params['image']
                tracked_feature = featuredata[result['id']]

                translation, rotation, draw_params = image_processing.solvePNPStereo(img1,
                                                                                     sift_params['keypoints'],
                                                                                     LANDMARKS[label_name]['good_matches_threshold'],
                                                                                     tracked_feature["left_keypoints"],
                                                                                     tracked_feature["left_good_matches"],
                                                                                     tracked_feature["right_keypoints"],
                                                                                     tracked_feature["right_good_matches"])

                ppm = LANDMARKS[results["label"]]['width'] / (result['x_max'] - result['x_min'])

                translation_m = translation * ppm

                yaw, ptch, roll = image_processing.decomposeYawPitchRoll(rotation)
                rotation_deg = 57.2958 * yaw

                # Avoid mirrors (https://github.com/opencv/opencv/issues/8813#issuecomment-359079875)
                if translation_m[0].item() < 0:
                    x_pos = -translation_m[0].item()
                elif translation_m[0].item() > FIELD_WIDTH:
                    x_pos = FIELD_WIDTH - (translation_m[0].item() - FIELD_WIDTH)
                else:
                    x_pos = translation_m[0].item()

                if translation_m[0].item() < 0:
                    y_pos = -translation_m[1].item()
                elif translation_m[1].item() > FIELD_HEIGHT:
                    y_pos = FIELD_HEIGHT - (translation_m[1].item() - FIELD_HEIGHT)
                else:
                    y_pos = translation_m[1].item()

                self.robot_pose3d = (x_pos, y_pos, result['depth_z'], 0)

            except Exception as e:
                log.warning("Exception caught doing solvePNP: {}".format(e))

        if len(results) == 0:
            self.has_targets = False

        print("FPS: {}".format(self.fps.tick_fps('frame')))
        self.nt_client.smartdashboard.putBoolean("has_targets", self.has_targets)
        self.nt_client.smartdashboard.putNumberArray("robot_pose", to_wpilib_coords(self.robot_pose3d))

        return results

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results, featuredata in depthai_utils.capture():
                self.parse_frame(frame, results, featuredata)

        finally:
            # depthai_utils.del_pipeline()
            pass


class MainDebug(Main):
    distance_class = DistanceCalculationsDebug
    field2d = Field2dWindow()

    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7

    def __init__(self):
        super().__init__()

    def parse_frame(self, frame, results, featuredata):
        # distance_results = super().parse_frame(frame, results)

        for result in results:
            if result['label'] != 5:
                continue

            try:
                label_name = depthai_utils.LABELS[result['label']]
                sift_params = SIFT_PARAMS[label_name]
                img1 = sift_params['image'].copy()
                tracked_feature = featuredata[result['id']]

                retval, translation, rotation, draw_params = image_processing.solvePNPStereo(img1,
                                                                                             sift_params['keypoints'],
                                                                                             LANDMARKS[label_name]['good_matches_threshold'],
                                                                                             tracked_feature["left_keypoints"],
                                                                                             tracked_feature["left_good_matches"],
                                                                                             tracked_feature["right_keypoints"],
                                                                                             tracked_feature["right_good_matches"])

                if not retval:
                    continue

                ppm = LANDMARKS[label_name]['width'] / (result['x_max'] - result['x_min'])

                translation_m = translation * ppm
                yaw, pitch, roll = image_processing.decomposeYawPitchRoll(rotation)
                rotation_deg = 57.2958 * yaw

                # Avoid mirrors (https://github.com/opencv/opencv/issues/8813#issuecomment-359079875)
                if translation_m[0].item() < 0:
                    x_pos = -translation_m[0].item()
                elif translation_m[0].item() > FIELD_WIDTH:
                    x_pos = FIELD_WIDTH - (translation_m[0].item() - FIELD_WIDTH)
                else:
                    x_pos = translation_m[0].item()

                if translation_m[0].item() < 0:
                    y_pos = -translation_m[1].item()
                elif translation_m[1].item() > FIELD_HEIGHT:
                    y_pos = FIELD_HEIGHT - (translation_m[1].item() - FIELD_HEIGHT)
                else:
                    y_pos = translation_m[1].item()

                self.robot_pose3d = (x_pos, y_pos, result['depth_z'], 0)

                cv2.putText(img1, "x: {}".format(translation_m[0].round(2)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "y: {}".format(translation_m[1].round(2)), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "z: {}".format(translation_m[2].round(2)), (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "r: {}".format(rotation_deg[0].round(2)), (0, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            except Exception as e:
                log.warning("Exception caught doing solvePNP: {}".format(e))

            try:
                h, w, d = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                if draw_params['left_draw_params']:
                    dst = cv2.perspectiveTransform(pts, draw_params['left_M'])
                    img2 = cv2.polylines(tracked_feature['leftFrame'], [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    img3 = cv2.drawMatches(img1, sift_params['keypoints'], img2, tracked_feature['left_keypoints'], tracked_feature['left_good_matches'], None, **draw_params['left_draw_params'])
                    cv2.imshow("Ransac", img3)

                if draw_params['right_draw_params']:
                    dst = cv2.perspectiveTransform(pts, draw_params['right_M'])
                    img2 = cv2.polylines(tracked_feature['rightFrame'], [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    img3 = cv2.drawMatches(img1, sift_params['keypoints'], img2, tracked_feature['right_keypoints'], tracked_feature['right_good_matches'], None, **draw_params['right_draw_params'])
                    cv2.imshow("Ransac", img3)
            except Exception as e:
                log.debug("Exception caught drawing solvePNP results: {}".format(e))

        print("Robot Pose: {}".format(self.robot_pose3d))
        field2d_frame = self.field2d.draw_robot_frame(self.robot_pose3d)

        cv2.imshow("Field", field2d_frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if args.debug:
        log.info("Setting up debug run...")
        MainDebug().run()
    else:
        log.info("Setting up non-debug run...")
        Main().run()