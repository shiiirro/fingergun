import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class LandmarkerWrapper:
    def __init__(self):
        self.result: HandLandmarkerResult | None = None
        self.landmarker: HandLandmarker | None = None
        self.create_landmarker()
        self.mode: str = "neutral"

    def create_landmarker(self):
        # callback function for mediapipe landmarker in livestream mode
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        self.landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = './hand_landmarker.task'),
            running_mode = VisionRunningMode.LIVE_STREAM,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            result_callback = callback,
            num_hands = 1,
        ))

    def update_from(self, frame: np.ndarray):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def annotated(self, frame: np.ndarray):
        if self.result is None or not self.result.hand_world_landmarks:
            return frame
        else:
            hand_landmarks_list = self.result.hand_landmarks
            annotated_frame = np.copy(frame)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y,
                                                    z = landmark.z)
                    for landmark in hand_landmarks])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            return annotated_frame

    def close(self):
        self.landmarker.close()
