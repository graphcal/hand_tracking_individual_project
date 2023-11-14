import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

"""
Variables for hand gesture control
"""
# pyautogui
screen_width, screen_height = pyautogui.size()
# s_w, s_h = 2560, 1440
print(f'screen width: {screen_width}, height: {screen_height}')

# For webcam input:
cap = cv2.VideoCapture(0)
# video capture width and height
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# w, h = 640, 480
print(f'video capture width: {w}, height: {h}')
print(f'screen width: {screen_width}, height: {screen_height}')

output_flip_code = 1

width_factor = screen_width / w
height_factor = screen_height / h

# To display FPS
time_prev = 0
time_curr = 0

mode = None
active = False

# 0: thumb, 1: index, 2: middle, 3: ring, 4: pinky, 5: message
gesture_index = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
gestures = [
    [False, True, False, False, False, "1"],
    [False, True, True, False, False, "2"],
    [False, True, True, True, False, "3"],
    [False, True, True, True, True, "4"],
    [True, True, True, True, True, "5"],
    [True, False, False, False, True, "6"]
]
gesture_open = [False, False, False, False, False]

"""
Functions for hand gesture control
"""


def _get_landmark(results):
    # convert output of mediapipe to a 21*3 matrix
    # hand pose is described by 21 key points
    # each point's position is described by a 3-dimensional vector (x, y, z)
    result_lm = results.multi_hand_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h, p.z * w] for p in result_lm])


def _get_landmarks(results):
    # improved _get_landmark for distinguishing left and right hand
    lm_results = {}
    if not results.multi_hand_landmarks:
        return lm_results
    for idx, classification in enumerate(results.multi_handedness):
        label = classification.classification[0].label
        lm_result = np.array([[p.x * w, p.y * h]
                              for p in results.multi_hand_landmarks[idx].landmark])
        # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models
        lm_results[label] = lm_result
    return lm_results


def _get_distance(lm0_x, lm0_y, lm1_x, lm1_y):
    return np.sqrt((lm0_x - lm1_x) ** 2 + (lm0_y - lm1_y) ** 2)


# "Left" or "Right" in _get_landmarks(results)


def draw_item(pos, frame, color=(0, 0, 255)):
    cv2.circle(frame, pos, 10, color, thickness=-1, lineType=cv2.FILLED)


def _get_distance_between_landmarks(landmarks_list, target_landmark):
    # lm_lists = _get_landmarks(results)
    # target_lm = [["Left", 5], ["Left", 8]]
    target_hand = {target_landmark[0][0], target_landmark[1][0]}
    target0_hand, target0_lm = target_landmark[0][0], target_landmark[0][1]
    target1_hand, target1_lm = target_landmark[1][0], target_landmark[1][1]

    lm0 = landmarks_list[target0_hand]
    lm0_x, lm0_y = lm0[target0_lm][0], lm0[target0_lm][1]
    lm1 = landmarks_list[target1_hand]
    lm1_x, lm1_y = lm1[target1_lm][0], lm1[target1_lm][1]

    distance = _get_distance(lm0_x, lm0_y, lm1_x, lm1_y)
    return distance


def is_touched(landmarks_list, target_landmark, threshold=10, debug=False):
    # lm_lists = _get_landmarks(results)
    # target_lm = [["Left", 5], ["Left", 8]]
    target_hand = {target_landmark[0][0], target_landmark[1][0]}
    if list(landmarks_list.keys()) != list(target_hand):
        return False
    else:
        target0_hand, target0_lm = target_landmark[0][0], target_landmark[0][1]
        target1_hand, target1_lm = target_landmark[1][0], target_landmark[1][1]

        lm0 = landmarks_list[target0_hand]
        lm0_x, lm0_y = lm0[target0_lm][0], lm0[target0_lm][1]
        lm1 = landmarks_list[target1_hand]
        lm1_x, lm1_y = lm1[target1_lm][0], lm1[target1_lm][1]

        distance = _get_distance(lm0_x, lm0_y, lm1_x, lm1_y)
        np.sqrt((lm0_x - lm1_x) ** 2 + (lm0_y - lm1_y) ** 2)
        if distance > threshold:
            return False
        else:
            if debug:
                print(
                    f'distance: {distance:.2f} ({target_landmark[0]},{target_landmark[1]})')
            return True


"""
Main section for hand gesture control
"""

with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, output_flip_code)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        # To improve performance, optionally mark the image as not writeable by passing reference.
        # image.flags.writeable = False

        # RGB -> BGR to show with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render hand detection results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style()
                                          )
            # results of landmarks into dictionary for easy access
            lm_lists = _get_landmarks(results)

            """
            functionalities of hand gesture control
            """
            # Left hand to switch modes
            if 'Left' in lm_lists:
                if 'Right' not in lm_lists:
                    # check if THUMB_TIP and other TIPs are close
                    if is_touched(lm_lists, [["Left", 4], ["Left", 8]], 20, debug=True):
                        mode = 'Cursor'
                        pyautogui.moveTo(100, 100)
                        gesture_open = [False, False, False, False, False]
                    elif is_touched(lm_lists, [["Left", 4], ["Left", 12]], 20, debug=True):
                        mode = 'Scroll'
                        pyautogui.moveTo(100, 100)
                        gesture_open = [False, False, False, False, False]
                    elif is_touched(lm_lists, [["Left", 4], ["Left", 16]], 20, debug=True):
                        mode = 'Gesture'
                        gesture_open = [False, False, False, False, False]
                    elif is_touched(lm_lists, [["Left", 4], ["Left", 20]], 20, debug=True):
                        mode = 'Touch'
                    else:
                        pass

            # Right hand to control
            if 'Right' in lm_lists:
                if 'Left' not in lm_lists:
                    for i in range(0, 5):
                        d1 = _get_distance_between_landmarks(
                            lm_lists,
                            [["Right", 0], ["Right", gesture_index[i][0]]]
                        )
                        d2 = _get_distance_between_landmarks(
                            lm_lists,
                            [["Right", 0], ["Right", gesture_index[i][1]]]
                        )
                        gesture_open[i] = d1 < d2
                    # print(gesture_open)
                    match mode:
                        # implement cursor
                        case 'Cursor':
                            print(gesture_open)
                            # 480:x=2560:1440, x=270
                            a_x, a_y = (80, 95)  # 640-160, 480
                            b_x, b_y = (560, 385)  # 460-270
                            cv2.rectangle(image, (a_x, a_y),
                                          (b_x, b_y), (0, 0, 0), 3)
                            right_index_x, right_index_y = lm_lists['Right'][8]
                            coef = w/(w-a_x*2)
                            cur_x = (right_index_x - a_x) * width_factor * coef
                            cur_y = (right_index_y - a_y) * \
                                height_factor * coef
                            if a_x <= right_index_x <= b_x and a_y <= right_index_y <= b_y:
                                pyautogui.moveTo(cur_x, cur_y)
                            if is_touched(lm_lists, [["Right", 4], ["Right", 5]], 10, debug=True):
                                pyautogui.click(button='left')
                            elif gesture_open == [True, True, True, True, True]:
                                mode = 'Scroll'
                        # implement scroll
                        case 'Scroll':
                            print(gesture_open)
                            gesture_scroll = [
                                [False, True, False, False, False],
                                [True, True, False, False, False],
                            ]
                            if gesture_open in gesture_scroll:
                                # if pointing upward
                                scroll_speed = 50
                                if lm_lists["Right"][0][1] < lm_lists["Right"][8][1]:
                                    pyautogui.scroll(-scroll_speed)
                                else:
                                    pyautogui.scroll(scroll_speed)
                            elif gesture_open == [False, False, False, False, False]:
                                mode = 'Cursor'
                        # implement gesture identification
                        case 'Gesture':
                            for i in range(0, len(gestures)):
                                flag = True
                                for j in range(0, 5):
                                    if gesture_open[j] != gestures[i][j]:
                                        flag = False
                                        break
                                if (flag == True):
                                    print(gestures[i][5])
                                    cv2.putText(image, gestures[i][5], (20, 160),
                                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
                        case 'Touch':
                            r_4_8 = [["Right", 4], ["Right",  8]]
                            r_4_12 = [["Right", 4], ["Right", 12]]
                            r_4_16 = [["Right", 4], ["Right", 16]]
                            r_4_20 = [["Right", 4], ["Right", 20]]
                            a1 = is_touched(lm_lists, r_4_8, 20, debug=True)
                            a2 = is_touched(lm_lists, r_4_12, 20, debug=True)
                            a3 = is_touched(lm_lists, r_4_16, 20, debug=True)
                            a4 = is_touched(lm_lists, r_4_20, 20, debug=True)
                            if a1 == True:
                                cv2.putText(image, str(r_4_8), (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            elif a2 == True:
                                cv2.putText(image, str(r_4_12), (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            elif a3 == True:
                                cv2.putText(image, str(r_4_16), (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            elif a4 == True:
                                cv2.putText(image, str(r_4_20), (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        case _:
                            pass
        time_curr = time.time()
        fps = 1 / (time_curr - time_prev)
        time_prev = time_curr

        cv2.putText(
            image, f"FPS: {int(fps)}", (20, 20),
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1
        )
        if mode is not None:
            cv2.putText(
                image, f"Mode: {mode}", (20, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2
            )

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
