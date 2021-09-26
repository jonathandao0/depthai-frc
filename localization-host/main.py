import argparse
import logging
import socket

import cv2
import depthai_utils
import numpy as np

from common import image_processing
from common.camera_info import OAK_L_PARAMS
from common.config import MODEL_NAME
from common.frame2dwindow import Frame2dWindow
from distance import DistanceCalculations, DistanceCalculationsDebug
from common.field_constants import *

from common.image_processing import SIFT_PARAMS
from common.utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:
    distance_class = DistanceCalculations
    # nt_class = NetworkTablesClient

    labels = None

    robot_pose3d = None
    has_targets = False

    def __init__(self):
        # self.nt_client = self.nt_class("localhost", "Depthai")
        self.pipeline, self.labels = depthai_utils.create_pipeline(MODEL_NAME)
        self.fps = FPSHandler()

        self.robot_pose3d = [FIELD_WIDTH / 2, 0, FIELD_HEIGHT / 2, 0]
        self.last_pose_update_time = 0

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))

        ip_address = s.getsockname()[0]
        # self.output_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=4201)

        image_processing.initalizeAllRefTargets()

    def parse_frame(self, results):
        # distance_results = self.distance_class.parse_frame(frame, results)
        for result in results['bboxes']:
            label_name = depthai_utils.LABELS[result['label']]

            if label_name == 'power_cell':
                continue

            self.has_targets = True

            try:
                tracked_feature = results['featuredata'][result['id']]

                ppm = LANDMARKS[label_name]['width'] / (result['x_max'] - result['x_min'])

                left_points_2d = np.array([kp.pt for kp in tracked_feature['left_keypoints']])
                right_points_2d = np.array([kp.pt for kp in tracked_feature['right_keypoints']])

                if np.shape(left_points_2d)[0] < 4:
                    log.debug("Not enough points for solvePNP ({})".format(len(left_points_2d)))
                    continue

                triangulated_points_4d = cv2.triangulatePoints(OAK_L_PARAMS['l_projection'],
                                                               OAK_L_PARAMS['r_projection'], left_points_2d.T * ppm,
                                                               right_points_2d.T * ppm)
                triangulated_points_3d = []
                for point in triangulated_points_4d.T:
                    triangulated_points_3d.append([point[0], point[1], point[2]])
                triangulated_points_3d = np.array(triangulated_points_3d)

                retval, rVec, tVec, _ = cv2.solvePnPRansac(triangulated_points_3d,
                                                           left_points_2d,
                                                           OAK_L_PARAMS['l_intrinsic'],
                                                           OAK_L_PARAMS['l_distortion'])

                if not retval:
                    continue

                rMat = cv2.Rodrigues(rVec)[0]

                yaw, pitch, roll = image_processing.decomposeYawPitchRoll(rMat.T)
                rotation_deg = 57.2958 * -yaw

                landmark_pos = LANDMARKS[label_name]['pose']
                object_pos = np.array([landmark_pos[0], landmark_pos[2], landmark_pos[1]]).reshape((3, 1))
                camera_pos = np.matmul(rMat.T, object_pos - tVec)

                x_pos = camera_pos[0][0]
                y_pos = camera_pos[1][0]
                z_pos = camera_pos[2][0]

                # Avoid mirrors (https://github.com/opencv/opencv/issues/8813#issuecomment-359079875)
                if x_pos < 0:
                    x_pos = -x_pos
                elif x_pos > FIELD_WIDTH:
                    x_pos = x_pos - FIELD_WIDTH
                if z_pos < 0:
                    z_pos = -z_pos
                elif z_pos > FIELD_HEIGHT:
                    z_pos = z_pos - FIELD_HEIGHT

                self.robot_pose3d = (x_pos, y_pos, z_pos, landmark_pos[3] - rotation_deg[0])

            except Exception as e:
                log.warning("Exception caught doing solvePNP: {}".format(e))

        if len(results) == 0:
            self.has_targets = False

        # self.nt_client.smartdashboard.putBoolean("has_targets", self.has_targets)
        # self.nt_client.smartdashboard.putNumberArray("robot_pose", pose3d_to_field2d(self.robot_pose3d))

        return results

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for results in depthai_utils.capture():
                self.parse_frame(results)

                # self.output_stream.sendFrame(results['frame'])

        finally:
            # depthai_utils.del_pipeline()
            pass


class MainDebug(Main):
    distance_class = DistanceCalculationsDebug
    field2d = Frame2dWindow()

    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7

    def __init__(self):
        super().__init__()

    def parse_frame(self, results):
        # distance_results = super().parse_frame(frame, results)

        for result in results['bboxes']:
            label_name = depthai_utils.LABELS[result['label']]

            if label_name == 'power_cell':
                continue

            if 'featuredata' not in results:
                continue

            try:
                label_name = depthai_utils.LABELS[result['label']]
                sift_params = SIFT_PARAMS[label_name]
                img1 = sift_params['image'].copy()
                tracked_feature = results['featuredata'][result['id']]

                # TODO: This calculation is wrong
                ppm = LANDMARKS[label_name]['width'] / (result['x_max'] - result['x_min'])
                # ppm = 1

                left_points_2d = np.array([kp.pt for kp in tracked_feature['left_keypoints']])
                right_points_2d = np.array([kp.pt for kp in tracked_feature['right_keypoints']])

                if np.shape(left_points_2d)[0] < 4:
                    log.info("Not enough points for solvePNP ({})".format(len(left_points_2d) ))
                    continue

                triangulated_points_4d = cv2.triangulatePoints(OAK_L_PARAMS['l_projection'], OAK_L_PARAMS['r_projection'], left_points_2d.T * ppm, right_points_2d.T * ppm)
                triangulated_points_3d = []
                for point in triangulated_points_4d.T:
                    triangulated_points_3d.append([point[0], point[1], point[2]])
                triangulated_points_3d = np.array(triangulated_points_3d)

                retval, rVec, tVec, _ = cv2.solvePnPRansac(triangulated_points_3d,
                                                           left_points_2d,
                                                           OAK_L_PARAMS['l_intrinsic'],
                                                           OAK_L_PARAMS['l_distortion'])

                if not retval:
                    continue

                rMat = cv2.Rodrigues(rVec)[0]

                yaw, pitch, roll = image_processing.decomposeYawPitchRoll(rMat.T)
                rotation_deg = 57.2958 * -yaw

                landmark_pos = LANDMARKS[label_name]['pose']
                object_pos = np.array([landmark_pos[0], landmark_pos[2], landmark_pos[1]]).reshape((3, 1))
                camera_pos = np.matmul(rMat.T, object_pos - tVec)

                x_pos = -camera_pos[0][0]
                y_pos = -camera_pos[1][0]
                z_pos = camera_pos[2][0]

                # Avoid mirrors (https://github.com/opencv/opencv/issues/8813#issuecomment-359079875)
                # if x_pos < 0:
                #     x_pos = -x_pos
                # elif x_pos > FIELD_WIDTH:
                #     x_pos = x_pos - FIELD_WIDTH
                # if z_pos < 0:
                #     z_pos = -z_pos
                # elif z_pos > FIELD_HEIGHT:
                #     z_pos = z_pos - FIELD_HEIGHT

                self.robot_pose3d = (x_pos, y_pos, z_pos, landmark_pos[3] - rotation_deg[0])

                cv2.putText(img1, "x: {}".format(x_pos.round(2)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "y: {}".format(y_pos.round(2)), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "z: {}".format(z_pos.round(2)), (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(img1, "r: {}".format(rotation_deg[0].round(2)), (0, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            except Exception as e:
                log.warning("Exception caught doing solvePNP: {}".format(e))

            try:
                h, w, d = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                if tracked_feature['draw_params']:
                    dst = cv2.perspectiveTransform(pts, tracked_feature['homography'])
                    img2 = cv2.polylines(tracked_feature['frame'], [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    img3 = cv2.drawMatches(img1, sift_params['keypoints'], img2, tracked_feature['keypoints'], tracked_feature['good_matches'], None, **tracked_feature['draw_params'])
                    cv2.imshow("Ransac", img3)

            except Exception as e:
                log.debug("Exception caught drawing solvePNP results: {}".format(e))

        print("Robot Pose: {}".format(self.robot_pose3d))
        field2d_frame = self.field2d.draw_robot(self.robot_pose3d)

        cv2.imshow("Field", field2d_frame)
        cv2.imshow("Frame", results['frame'])
        # if disparityFrame.size != 0:
        #     cv2.imshow("Disparity", disparityFrame)

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