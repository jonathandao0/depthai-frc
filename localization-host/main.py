import logging

import cv2
import numpy as np

from common.config import MODEL_NAME, DEBUG, NN_IMG_SIZE, DETECTION_PADDING
from common.depthai_utils import DepthAI, DepthAIDebug
from distance import DistanceCalculations, DistanceCalculationsDebug
from common.field_constants import *

from common.networktables_client import NetworkTablesClient
from process_image import homography_to_perspective_transform
from common.camera_info import CAMERA_RGB
from utils import FPSHandler

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI
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
        self.depthai = self.depthai_class(MODEL_NAME)
        self.labels = self.depthai.labels
        self.distance = self.distance_class()
        self.fps = FPSHandler()

        self.robot_pose3d = [FIELD_HEIGHT / 2, FIELD_WIDTH / 2, 0, 0]

        for name, landmark in LANDMARKS.items():
            try:
                img = cv2.imread("resources/images/{}.jpg".format(name))
                # keypoints, descriptors = self.orb.detectAndCompute(img, None)
                # landmark["orb_params"] = {
                keypoints, descriptors = self.sift.detectAndCompute(img, None)
                landmark["sift_params"] = {
                    'image': img,
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }
            except Exception:
                pass

    def parse_frame(self, frame, results):
        self.fps.tick('frame')
        distance_results = self.distance.parse_frame(frame, results)
        for result in results:
            self.has_targets = True

            if result['label'] == 5 or result['label'] == 4:
                self.robot_pose3d = LANDMARKS['red_upper_power_port_sandbox']['pose']
            elif result['label'] == 0 or result['label'] == 1:
                self.robot_pose3d = LANDMARKS['blue_upper_power_port_sandbox']['pose']
            elif result['label'] == 3:
                self.robot_pose3d = LANDMARKS['red_loading_bay_sandbox']['pose']
            # if result['label'] == 6:
            #     self.robot_position = LANDMARKS['blue_loading_bay_sandbox']['pose']

        if len(results) == 0:
            self.robot_pose3d = (-99, -99, 0)
            self.has_targets = False

        print("Robot Position: {}".format(self.robot_pose3d))
        print("FPS: {}".format(self.fps.tick_fps('frame')))
        self.nt_client.smartdashboard.putBoolean("has_targets", self.has_targets)
        self.nt_client.smartdashboard.putNumberArray("robot_pose", to_wpilib_coords(self.robot_pose3d))

        return distance_results

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results in self.depthai.capture():
                self.parse_frame(frame, results)

        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug
    distance_class = DistanceCalculationsDebug
    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7

    scaled_values = None
    red_power_port = None
    blue_power_port = None
    red_loading_bay = None
    blue_loading_bay = None
    red_station_1 = None
    red_station_2 = None
    red_station_3 = None
    blue_station_1 = None
    blue_station_2 = None
    blue_station_3 = None

    def __init__(self):
        super().__init__()
        self.field_frame = self.make_field_image()

    def make_field_image(self):
        fov = 68.7938
        min_distance = 0.827

        frame = cv2.imread("resources/images/frc2020fieldOfficial.png")
        height, width = frame.shape[:2]

        self.scaled_values = (width / FIELD_WIDTH, height / FIELD_HEIGHT)

        red_power_port = pose3d_to_frame_position(LANDMARKS['red_upper_power_port_sandbox']['pose'], self.scaled_values)
        blue_power_port = pose3d_to_frame_position(LANDMARKS['blue_upper_power_port_sandbox']['pose'], self.scaled_values)
        red_loading_bay = pose3d_to_frame_position(LANDMARKS['red_loading_bay']['pose'], self.scaled_values)
        blue_loading_bay = pose3d_to_frame_position(LANDMARKS['blue_loading_bay']['pose'], self.scaled_values)

        red_station_1 = pose3d_to_frame_position(LANDMARKS['red_station_1']['pose'], self.scaled_values)
        red_station_2 = pose3d_to_frame_position(LANDMARKS['red_station_2']['pose'], self.scaled_values)
        red_station_3 = pose3d_to_frame_position(LANDMARKS['red_station_3']['pose'], self.scaled_values)

        blue_station_1 = pose3d_to_frame_position(LANDMARKS['blue_station_1']['pose'], self.scaled_values)
        blue_station_2 = pose3d_to_frame_position(LANDMARKS['blue_station_2']['pose'], self.scaled_values)
        blue_station_3 = pose3d_to_frame_position(LANDMARKS['blue_station_3']['pose'], self.scaled_values)

        cv2.circle(frame, red_power_port, 5, (0, 0, 255))
        cv2.circle(frame, blue_power_port, 5, (255, 0, 0))
        cv2.circle(frame, red_loading_bay, 5, (0, 0, 255))
        cv2.circle(frame, blue_loading_bay, 5, (255, 0, 0))

        cv2.circle(frame, red_station_1, 5, (0, 0, 255))
        cv2.circle(frame, red_station_2, 5, (0, 0, 255))
        cv2.circle(frame, red_station_3, 5, (0, 0, 255))

        cv2.circle(frame, blue_station_1, 5, (255, 0, 0))
        cv2.circle(frame, blue_station_2, 5, (255, 0, 0))
        cv2.circle(frame, blue_station_3, 5, (255, 0, 0))

        return frame

    def calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.field_frame.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.field_frame.shape[1])
        return int(bottom_x), int(top_x)

    def calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.field_frame.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.field_frame.shape[0])
        return int(bottom_z), int(top_z)

    def draw_robot(self, frame):
        try:
            r_left_top, r_right_bottom = robot_pose_to_frame_position(self.robot_pose3d, self.scaled_values)

            cv2.rectangle(frame, r_left_top, r_right_bottom, (0, 255, 0), -1)
        except Exception:
            pass

    def parse_frame(self, frame, results):
        distance_results = super().parse_frame(frame, results)

        field_frame = self.field_frame.copy()
        too_close_ids = []
        for result in distance_results:
            if result['dangerous']:
                left, right = self.calc_x(result['detection1']['depth_x'])
                top, bottom = self.calc_z(result['detection1']['depth_z'])
                cv2.rectangle(field_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection1']['id'])
                left, right = self.calc_x(result['detection2']['depth_x'])
                top, bottom = self.calc_z(result['detection2']['depth_z'])
                cv2.rectangle(field_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection2']['id'])

        for result in results:
            x1 = 0 if result['x_min'] - DETECTION_PADDING < 0 else result['x_min'] - DETECTION_PADDING
            x2 = NN_IMG_SIZE if result['x_max'] + DETECTION_PADDING < NN_IMG_SIZE \
                else result['x_max'] + DETECTION_PADDING
            y1 = 0 if result['y_min'] - DETECTION_PADDING < 0 else result['y_min'] - DETECTION_PADDING
            y2 = NN_IMG_SIZE if result['y_max'] + DETECTION_PADDING < NN_IMG_SIZE \
                else result['y_max'] + DETECTION_PADDING

            cropped_frame = frame[y1:y2, x1:x2]

            if result['label'] == 5:
                detection_params = LANDMARKS['red_upper_power_port_sandbox']['sift_params']
                source_keypoints = detection_params['keypoints']
                target_keypoints, target_descriptors = self.sift.detectAndCompute(cropped_frame, None)
                matches = self.flann.knnMatch(detection_params['descriptors'], target_descriptors, k=2)

                good_matches = []

                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                # detection_params = LANDMARKS['red_upper_power_port_sandbox']['orb_params']
                # source_keypoints = orb_params['keypoints']
                # target_keypoints, target_descriptors = self.orb.detectAndCompute(cropped_frame, None)
                # matches = self.bf.match(orb_params['descriptors'], target_descriptors)
                #
                # good_matches = sorted(matches, key=lambda x: x.distance)

                if len(good_matches) > 10:
                    try:
                        src_pts = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        corner_camera_coord, object_points_3d, center_pts = homography_to_perspective_transform(detection_params['image'].shape, M)

                        corner_camera_coord = corner_camera_coord.reshape(-1, 2)
                        # solve pnp using iterative LMA algorithm
                        retval, rotation_rad, tMatrix = cv2.solvePnP(object_points_3d, corner_camera_coord,
                                                                             CAMERA_RGB['intrinsicMatrix'],
                                                                             CAMERA_RGB['distortionCoeff'])

                        rotM = cv2.Rodrigues(rotation_rad)[0]
                        translation = -np.matrix(rotM).T * np.matrix(tMatrix)

                        H = M
                        K = CAMERA_RGB['intrinsicMatrix']
                        h1 = H[0]
                        h2 = H[1]
                        h3 = H[2]
                        K_inv = np.linalg.inv(K)
                        L = 1 / np.linalg.norm(np.dot(K_inv, h1))
                        r1 = L * np.dot(K_inv, h1)
                        r2 = L * np.dot(K_inv, h2)
                        r3 = np.cross(r1, r2)
                        T = L * (K_inv @ h3.reshape(3, 1))
                        R = np.array([[r1], [r2], [r3]])
                        R = np.reshape(R, (3, 3))

                        print("X: {}\tY: {}\tZ: {}".format(T[0], T[1], T[2]))
                        matchesMask = mask.ravel().tolist()
                        img1 = detection_params['image'].copy()
                        h, w, d = img1.shape

                        ppm = LANDMARKS['red_upper_power_port_sandbox']['width'] / (x2 - x1)

                        translation = ppm * translation
                        rotation_deg = 57.2958 * rotation_rad

                        self.robot_pose3d = (translation[0].item(), translation[1].item(), result['depth_z'], 0)
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        img2 = cv2.polylines(cropped_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                           singlePointColor=None,
                                           matchesMask=matchesMask,  # draw only inliers
                                           flags=2)
                        cv2.putText(img1, "x: {}".format(translation[0].round(2)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                        cv2.putText(img1, "y: {}".format(translation[1].round(2)), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                        cv2.putText(img1, "z: {}".format(round(result['depth_z'], 2)), (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                        cv2.putText(img1, "r: {}".format(rotation_deg[0].round(2)), (0, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                        img3 = cv2.drawMatches(img1, source_keypoints, img2, target_keypoints, good_matches, None, **draw_params)
                        cv2.imshow("Ransac", img3)
                    except Exception:
                        pass

        self.draw_robot(field_frame)

        cv2.imshow("Field", field_frame)
        cv2.imshow("Frame", frame)

        # test = self.depthai_class.get_frames(self)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if DEBUG:
        log.info("Setting up debug run...")
        MainDebug().run()
    else:
        log.info("Setting up non-debug run...")
        Main().run()