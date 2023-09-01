import cv2 as cv
from model import HandModel
import pydirectinput
import numpy as np
from collections import deque


# constants
sens = 15000
moving_average_window_size = 5
pydirectinput.FAILSAFE = False


def main():
    model = HandModel(debug = True)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened(): exit()

    window = deque(maxlen = moving_average_window_size)
    prev_idx_loc = []
    shooting = False
    while True:
        online, frame = cap.read()
        if not online: break
        frame = cv.flip(frame, 1)

        model.update_from(frame)
        frame = model.mark_on(frame)

        if model.raw_results.hand_landmarks and model.pose != "HOVER":
            window.append(model.coordinates[8])
            idx_loc = np.average(window, axis = 0)
            rel_move = (int((prev_idx_loc[2] - idx_loc[2]) * sens),
                        int((prev_idx_loc[1] - idx_loc[1]) * sens)) if len(prev_idx_loc) > 0 else (
                0, 0)

            # do things
            if model.pose == "SHOOT" and not shooting:
                pydirectinput.mouseDown(_pause = False)
                shooting = True
            elif model.pose != "SHOOT" and shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False
            if model.pose == "RELOAD":
                pydirectinput.press('r')

            pydirectinput.moveRel(*rel_move, relative = True, _pause = False, disable_mouse_acceleration = True)

            prev_idx_loc = idx_loc
        else:
            if shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False
            window.clear()
            prev_idx_loc = []

        cv.imshow('live', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    model.close()

if __name__ == "__main__":
    main()
