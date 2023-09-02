import cv2 as cv
from model import HandModel
import pydirectinput
import numpy as np
from collections import deque


pydirectinput.FAILSAFE = False

#######################################################
# constants/settings, modify as needed
PTR_SENS = 15000
MOVING_AVG_WINDOW_LEN = 5
COOLDOWN_MAX = 0    # set to 0 to disable
                    # disables mouse movement for a short time after transitioning to a SHOOT pose
                    # makes aiming at static targets easier but makes tracking more difficult
#######################################################

def main():
    model = HandModel(debug = True)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened(): exit()

    # moving average to smooth model coordinates stream
    window = deque(maxlen = MOVING_AVG_WINDOW_LEN)

    prev_ptr_coord = []
    shooting = False
    cooldown = 0

    while True:
        online, frame = cap.read()
        if not online: break
        frame = cv.flip(frame, 1)

        model.update_from(frame)
        frame = model.mark_on(frame)

        if model.raw_results.hand_landmarks and model.pose != "HOVER":
            # pose processing
            if model.pose == "SHOOT" and not shooting:
                pydirectinput.mouseDown(_pause = False)
                shooting = True
                cooldown = COOLDOWN_MAX
            elif model.pose != "SHOOT" and shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False

            if model.pose == "RELOAD":
                pydirectinput.press('r')

            if cooldown <= 0:
                window.append(model.coordinates[8])
            ptr_coord = np.average(window, axis = 0)

            # calculation of pixel movement from previous coordinates
            rel_move = (int((prev_ptr_coord[2] - ptr_coord[2]) * PTR_SENS),
                        int((prev_ptr_coord[1] - ptr_coord[1]) * PTR_SENS)) if len(
                prev_ptr_coord) > 0 else (
                0, 0)

            # relative mouse movement
            pydirectinput.moveRel(*rel_move, relative = True, _pause = False,
                                  disable_mouse_acceleration = True)

            prev_ptr_coord = ptr_coord
        else:
            window.clear()
            prev_ptr_coord = []
            if shooting:
                pydirectinput.mouseUp(_pause = False)
                shooting = False
            cooldown = 0

        cv.imshow('live', frame)
        cooldown -= 1

        # exit key
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    model.close()


if __name__ == "__main__":
    main()
