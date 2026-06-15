import cv2
import mediapipe as mp
import numpy as np
from predict_handwriting import predict_from_canvas
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
canvas=np.ones((480, 640, 3), dtype=np.uint8)*255
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
prev_x, prev_y=None, None
mode='DRAW'
current_word= ""
drawing_active= False
clear_counter= 0
no_hand_counter= 0
CLEAR_HOLD_FRAMES= 12
CLR_WHITE= (255, 255, 255)
CLR_BLACK= (0, 0, 0)
CLR_GREEN= (0, 255, 0)
CLR_RED= (0, 0, 255)
CLR_BLUE= (255, 0, 0)
CLR_GRAY= (180, 180, 180)
def get_fingers_state(hand_lms):
    tips= [8, 12, 16, 20]
    pips= [6, 10, 14, 18]
    fingers= []
    for tip, pip in zip(tips, pips):
        fingers.append(1 if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y else 0)
    return fingers
while True:
    success, frame= cap.read()
    if not success: break

    frame= cv2.flip(frame, 1)
    h, w, _= frame.shape
    rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result= hands.process(rgb)
    ui_y_limit= h-60
    cv2.rectangle(canvas, (0, ui_y_limit), (w,h), (40,40,40),-1)
    cv2.putText(canvas, f"STRING: {current_word}", (20,h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_WHITE, 2)
    pointer_pos= None
    pointer_color= CLR_RED
    if result.multi_hand_landmarks:
        no_hand_counter= 0
        hand_lms= result.multi_hand_landmarks[0]
        fingers= get_fingers_state(hand_lms)
        finger_count= sum(fingers)        
        ix, iy= int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)
        pointer_pos= (ix, iy)
        in_draw_button= (10 < ix < 140 and 10 < iy < 70)
        in_write_button= (160 < ix < 290 and 10 < iy < 70)
        if in_draw_button:
            mode= 'DRAW'
            prev_x, prev_y= None, None
        elif in_write_button:
            mode= 'WRITE'
            prev_x, prev_y= None, None
        elif fingers== [1, 1, 1, 1]:
            pointer_color= CLR_WHITE
            clear_counter+= 1
            if clear_counter>= CLEAR_HOLD_FRAMES:
                canvas[:ui_y_limit, :]= 255
                clear_counter= 0
            prev_x, prev_y= None, None
        elif fingers== [1, 1, 0, 0]:
            pointer_color= CLR_BLUE
            if prev_x is not None:
                smooth_x= int(0.7 * prev_x + 0.3* ix)
                smooth_y= int(0.7 * prev_y + 0.3* iy)
                cv2.line(canvas, (prev_x, prev_y),(smooth_x, smooth_y),CLR_WHITE, 25)
                prev_x, prev_y= smooth_x, smooth_y
            else:
                prev_x, prev_y= ix, iy
        elif fingers== [1, 0, 0, 0]:
            pointer_color= CLR_GREEN
            drawing_active= True
            if prev_x is not None:
                smooth_x= int(0.7 * prev_x + 0.3* ix)
                smooth_y= int(0.7 * prev_y + 0.3* iy)
                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), CLR_RED, 5)
                prev_x, prev_y= smooth_x, smooth_y
            else:
                prev_x, prev_y= ix, iy
        elif finger_count== 0:
            pointer_color= CLR_RED
            prev_x, prev_y= None, None
        
        else:
            prev_x, prev_y= None, None
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y= None, None
        no_hand_counter+= 1
        if mode== 'WRITE' and drawing_active and no_hand_counter> 15:
            char= predict_from_canvas(canvas[:ui_y_limit, :])
            if char:
                current_word+= char
                canvas[:ui_y_limit, :]= 255
            drawing_active= False
            no_hand_counter= 0
    camera_view= frame.copy()
    canvas_view= canvas.copy()
    cv2.rectangle(camera_view, (10, 10), (140, 70), (200, 200, 200), -1)
    cv2.putText(camera_view, "DRAW", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_BLACK, 2)
    cv2.rectangle(camera_view, (160, 10), (290, 70), (200, 200, 200), -1)
    cv2.putText(camera_view, "WRITE", (185, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_BLACK, 2)
    active_rect= (10, 10, 140, 70) if mode== 'DRAW' else (160, 10, 290, 70)
    cv2.rectangle(camera_view, (active_rect[0], active_rect[1]), (active_rect[2], active_rect[3]), CLR_GREEN, 3)
    cv2.putText(camera_view, "CAMERA", (w - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_GREEN, 2)
    cv2.putText(canvas_view, "CANVAS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_BLACK, 2)
    if pointer_pos:
        cv2.circle(camera_view, pointer_pos, 10, pointer_color, -1)
        cv2.circle(canvas_view, pointer_pos, 10, pointer_color, -1)
    combined= np.hstack((camera_view, canvas_view))
    cv2.imshow("SketchNova-Air Doodle", combined)
    key= cv2.waitKey(1) & 0xFF
    if key== ord('q'): break
    elif key== 32: current_word += " "
    elif key== 8: current_word = current_word[:-1]
cap.release()
cv2.destroyAllWindows()

