import cv2 as cv
from landmarker import LandmarkerWrapper
import numpy as np


def main():
    lm = LandmarkerWrapper()
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened(): exit()
    basis = None
    vec = None
    while True:
        # Read frame
        online, frame = cap.read()
        # Check frame status
        if not online: break
        # Flip frame
        frame = cv.flip(frame, 1)

        # Feed to mediapipe landmarker wrapper
        lm.update_from(frame)

        # print(lm.result)
        # print(lm.calc_basis())

        if lm.result is not None and len(lm.result.hand_landmarks) > 0:
            # pseudo camera internals
            frame_height, frame_width, channels = (480, 640, 3)
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype = "double"
            )
            distortion = np.zeros((4, 1))

            model_points = np.float32([[-l.x, -l.y, -l.z] for l in lm.result.hand_world_landmarks[0]])
            image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in lm.result.hand_landmarks[0]])
            success, rotation_vector, translation_vector = cv.solvePnP(model_points, image_points,
                                                                        camera_matrix, distortion,
                                                                        flags = cv.SOLVEPNP_SQPNP)

            # needs to be 4x4 because you have to use homogeneous coordinates
            transformation = np.eye(4)
            transformation[0:3, 3] = translation_vector.squeeze()
            # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

            # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis = 1)

            # apply the transformation
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)

            print(f'x: {world_points[8][0]}')
            print(f'y: {world_points[8][1]}')
            print(f'z: {world_points[8][2]}\n')

        # Draw landmarks on frame, if any
        frame = lm.annotated(frame)

        # Show frame
        cv.imshow('feed', frame)

        # Escape key
        if cv.waitKey(1) == ord('q'):
            break

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    lm.close()


if __name__ == "__main__":
    main()
