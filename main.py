import cv2 as cv
from model import HandModel
import pydirectinput
import numpy as np
from collections import deque


pydirectinput.FAILSAFE = False

# constants/settings, modify as needed
ptr_sens = 15000
moving_average_window_size = 5
shoot_movement_cd = 3 # TODO

def main():
    model = HandModel(debug = True)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened(): exit()

    # moving average to smooth model coordinates stream
    window = deque(maxlen = moving_average_window_size)

    prev_ptr_coord = []
    shooting = False

    while True:
        online, frame = cap.read()
        if not online: break
        frame = cv.flip(frame, 1)

        model.update_from(frame)
        frame = model.mark_on(frame)

        if model.raw_results.hand_landmarks and model.pose != "HOVER":
            window.append(model.coordinates[8])
            ptr_coord = np.average(window, axis = 0)

            # calculation of pixel movement from previous coordinates
            rel_move = (int((prev_ptr_coord[2] - ptr_coord[2]) * ptr_sens),
                        int((prev_ptr_coord[1] - ptr_coord[1]) * ptr_sens)) if len(
                prev_ptr_coord) > 0 else (
                0, 0)

            # pose processing
            if model.pose == "SHOOT" and not shooting:
                pydirectinput.mouseDown(_pause = False)
                shooting = True
            elif model.pose != "SHOOT" and shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False
            if model.pose == "RELOAD":
                pydirectinput.press('r')

            # relative mouse movement
            pydirectinput.moveRel(*rel_move, relative = True, _pause = False,
                                  disable_mouse_acceleration = True)

            prev_ptr_coord = ptr_coord
        else:
            if shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False
            window.clear()
            prev_ptr_coord = []

        cv.imshow('live', frame)

        # exit key
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    model.close()


if __name__ == "__main__":
    main()
