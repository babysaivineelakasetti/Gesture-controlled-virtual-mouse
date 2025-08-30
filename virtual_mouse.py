import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque

# --------------------- Settings ---------------------
VISUALIZE = True      # Set False for max speed
SHOW_FPS  = True
FRAME_W, FRAME_H = 640, 480
FRAME_MARGIN = 150    # larger margin = slower cursor
SMOOTH_ALPHA = 0.15   # smaller = smoother, slower cursor
GESTURE_COOLDOWN = 0.5    # seconds between gestures
IDLE_THRESHOLD = 5        # idle time
SCROLL_INTERVAL = 0.2     # slower scrolling
SCROLL_STEP = 30          # smaller scroll step
# ---------------------------------------------------

pyautogui.FAILSAFE = False
wScr, hScr = pyautogui.size()

def dist(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])

def map_to_screen(x, y, w, h):
    # map inner area (with margins) to full screen
    X = np.interp(x, [FRAME_MARGIN, w - FRAME_MARGIN], [0, wScr])
    Y = np.interp(y, [FRAME_MARGIN, h - FRAME_MARGIN], [0, hScr])
    return float(np.clip(X, 0, wScr-1)), float(np.clip(Y, 0, hScr-1))

def draw_text(img, text, pos=(20, 40), color=(0,255,0), scale=0.8):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# State
sx, sy = None, None
last_active = time.time()
last_action = 0.0
last_scroll_tick = 0.0
dragging = False
fps_clock = deque(maxlen=30)

while True:
    ok, img = cap.read()
    if not ok:
        continue

    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    now = time.time()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    if res.multi_hand_landmarks:
        last_active = now
        hand = res.multi_hand_landmarks[0]
        lm = hand.landmark

        # Important landmarks
        pts = {}
        for idx in (4,8,12,16,5,17,6,10):
            pts[idx] = (int(lm[idx].x*w), int(lm[idx].y*h))

        thumb = pts[4]; index = pts[8]; middle = pts[12]; ring = pts[16]
        index_mcp = pts[5]; pinky_mcp = pts[17]

        # Palm width scale
        palm_width = max(dist(index_mcp, pinky_mcp), 1.0)
        pinch_on = 0.45 * palm_width

        # Cursor position
        target_x, target_y = map_to_screen(index[0], index[1], w, h)

        # EMA smoothing
        if sx is None:
            sx, sy = target_x, target_y
        else:
            sx = SMOOTH_ALPHA * target_x + (1 - SMOOTH_ALPHA) * sx
            sy = SMOOTH_ALPHA * target_y + (1 - SMOOTH_ALPHA) * sy

        pyautogui.moveTo(sx, sy)

        # Finger states
        index_up  = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y

        # Distances
        d_thumb_index  = dist(thumb, index)
        d_thumb_middle = dist(thumb, middle)
        d_thumb_ring   = dist(thumb, ring)

        # Gestures (with cooldown)
        if (now - last_action) >= GESTURE_COOLDOWN:
            if d_thumb_index < pinch_on and not dragging:
                pyautogui.click()
                last_action = now
                if VISUALIZE: draw_text(img, "Left Click")

            elif d_thumb_middle < pinch_on and not dragging:
                pyautogui.click(button="right")
                last_action = now
                if VISUALIZE: draw_text(img, "Right Click", (20, 70), (0,0,255))

            elif d_thumb_ring < pinch_on and not dragging:
                pyautogui.doubleClick()
                last_action = now
                if VISUALIZE: draw_text(img, "Double Click", (20, 100), (255,0,255))

        # Dragging
        if index_up and middle_up and not dragging:
            pyautogui.mouseDown()
            dragging = True
        elif (not index_up or not middle_up) and dragging:
            pyautogui.mouseUp()
            dragging = False

        if VISUALIZE and dragging:
            draw_text(img, "Dragging", (20, 130), (255,255,0))

        # Scrolling
        if (index_up and not middle_up) or (middle_up and not index_up):
            if (now - last_scroll_tick) >= SCROLL_INTERVAL:
                direction = 1 if (index_up and not middle_up) else -1
                pyautogui.scroll(direction * SCROLL_STEP)
                last_scroll_tick = now
                if VISUALIZE:
                    draw_text(img, "Scroll Up" if direction>0 else "Scroll Down", (20, 160))

        if VISUALIZE:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(img, (FRAME_MARGIN, FRAME_MARGIN),
                          (w - FRAME_MARGIN, h - FRAME_MARGIN), (100, 100, 100), 1)

    else:
        if (now - last_active) > IDLE_THRESHOLD and VISUALIZE:
            draw_text(img, "Idle Mode - Battery Saver", (20, 40), (0,0,255))
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    # FPS
    fps_clock.append(time.time())
    if SHOW_FPS and len(fps_clock) >= 2:
        dt = fps_clock[-1] - fps_clock[0]
        fps = (len(fps_clock)-1) / dt if dt > 0 else 0
        if VISUALIZE: draw_text(img, f"FPS: {fps:.1f}", (w-160, 40), (255,255,255), 0.7)

    cv2.imshow("Virtual Mouse (Smooth)", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
