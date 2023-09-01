import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
import cv2 as cv
import numpy as np
import time


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# pseudo camera internals
f_height, f_width, channels = (480, 640, 3)
focal_len = f_width
center = (f_width / 2, f_height / 2)
camera_matrix = np.array(
    [[focal_len, 0, center[0]],
     [0, focal_len, center[1]],
     [0, 0, 1]], dtype = "double")
distortion = np.zeros((4, 1))


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class HandModel:
    def __init__(self, debug = False):
        self.raw_results = HandLandmarkerResult([], [], [])
        self.coordinates = np.empty((21, 3))
        self.pose = "HOVER"
        self._create_landmarker()

        self.debug = debug

    def update_from(self, frame: np.ndarray):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def mark_on(self, frame: np.ndarray):
        if self.raw_results.hand_landmarks:
            hands = self.raw_results.hand_landmarks
            annotated_frame = np.copy(frame)
            for i in range(len(hands)):
                landmarks = hands[i]
                proto = NormalizedLandmarkList()
                proto.landmark.extend(
                    [NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z)
                     for landmark in landmarks])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
            if self.debug:
                cv.putText(annotated_frame, self.pose, (50, 50), cv.FONT_HERSHEY_DUPLEX, 1.5,
                           (50, 50, 50), 2)
            return annotated_frame
        return frame

    def close(self):
        self.landmarker.close()

    def _create_landmarker(self):
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.raw_results = result
            self._calculate_3D_world_coordinates()
            self._determine_pose()

        self.landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = './hand_landmarker.task'),
            running_mode = VisionRunningMode.LIVE_STREAM,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            result_callback = callback,
            num_hands = 1,
        ))

    def _calculate_3D_world_coordinates(self):
        """Uses OpenCV PnP to estimate real world coordinates relative to camera"""
        if self.raw_results.hand_landmarks:
            model_points = np.float32(
                [[-l.x, -l.y, -l.z] for l in self.raw_results.hand_world_landmarks[0]])
            image_points = np.float32(
                [[l.x * f_width, l.y * f_height] for l in self.raw_results.hand_landmarks[0]])
            _, rotation_vec, translation_vec = cv.solvePnP(model_points, image_points,
                                                           camera_matrix, distortion,
                                                           flags = cv.SOLVEPNP_SQPNP)
            transformation = np.eye(4)
            transformation[0:3, 3] = translation_vec.squeeze()
            hom_coord = np.concatenate((model_points, np.ones((21, 1))), axis = 1)
            self.coordinates = np.delete(hom_coord.dot(np.linalg.inv(transformation).T), 3, 1)

    def _determine_pose(self):
        if self.raw_results.hand_landmarks:
            idx_mid_dist = np.linalg.norm(self.coordinates[8] - self.coordinates[11])
            if idx_mid_dist < 0.05:
                self.pose = "HOVER"
            else:
                thumb_idx_dist = min(np.linalg.norm(self.coordinates[4] - self.coordinates[5]), np.linalg.norm(self.coordinates[4] - self.coordinates[6]))
                if thumb_idx_dist < 0.04:
                    self.pose = "SHOOT"
                else:
                    hand_normal = np.cross(self.coordinates[5] - self.coordinates[0],
                                           self.coordinates[5] - self.coordinates[17])
                    flat_normal = np.array([0, 1, 0])

                    angle = angle_between(hand_normal, flat_normal)
                    print(angle)
                    if angle < 1.3:
                        self.pose = "RELOAD"
                    else:
                        self.pose = "DEFAULT"
        else:
            self.pose = "DEFAULT"