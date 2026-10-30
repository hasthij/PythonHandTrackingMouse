import sys
import subprocess

required_packages = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pyautogui": "pyautogui",
    "numpy": "numpy",
}

for module_name, pip_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"[INFO] Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import math
import os
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Configuration ----------------
FRAME_WIDTH = 320            # 160x120 is below what most webcams / MediaPipe track reliably
FRAME_HEIGHT = 240
DISPLAY_SCALE = 2
PINCH_THRESHOLD = 0.046
POSITION_SMOOTHING = 0.6     # 0..1, higher = more responsive, lower = smoother
TRAVEL = 0.4                 # fraction of frame the hand travels to cross the screen half-range
SCREEN_REACH = 0.7
HOLD_THRESHOLD = 0.35        # seconds of continuous pinch before "HOLD" is shown
STABLE_FRAMES = 2            # consecutive frames needed to change pinch state
CLICK_COOLDOWN = 0.2
CALIBRATION_FRAMES = 5
WINDOW = "Zero Delay Hand Control"

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
screen_width, screen_height = pyautogui.size()

# ---------------- State ----------------
initial_thumb_pos = None
smoothed = None
thumb_history = []
calibrating = True

left_down = False            # physical mouse button state
left_start = 0.0
last_click_time = 0.0
right_latched = False

left_state = False           # debounced pinch states
right_state = False
left_count = 0               # consecutive frames disagreeing with current state
right_count = 0

processing_active = threading.Event()
processing_active.set()
frame_lock = threading.Lock()
new_frame_ready = threading.Event()
current_frame = None
camera_error = threading.Event()


def dist(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def debounce(raw, state, count):
    """Flip state only after STABLE_FRAMES consecutive opposing readings."""
    if raw == state:
        return state, 0
    count += 1
    if count >= STABLE_FRAMES:
        return raw, 0
    return state, count


def capture_frames():
    global current_frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        camera_error.set()
        processing_active.clear()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    while processing_active.is_set():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            with frame_lock:
                current_frame = frame
            new_frame_ready.set()
        else:
            time.sleep(0.005)
    cap.release()


def release_mouse():
    global left_down
    if left_down:
        pyautogui.mouseUp()
        left_down = False


def move_cursor(nx, ny):
    global smoothed
    if smoothed is None:
        smoothed = (nx, ny)
    else:
        smoothed = (smoothed[0] + POSITION_SMOOTHING * (nx - smoothed[0]),
                    smoothed[1] + POSITION_SMOOTHING * (ny - smoothed[1]))
    rel_x = (smoothed[0] - initial_thumb_pos[0]) / TRAVEL
    rel_y = (smoothed[1] - initial_thumb_pos[1]) / TRAVEL
    sx = min(max(0, screen_width / 2 + rel_x * screen_width * SCREEN_REACH), screen_width - 1)
    sy = min(max(0, screen_height / 2 + rel_y * screen_height * SCREEN_REACH), screen_height - 1)
    pyautogui.moveTo(int(sx), int(sy))


def handle_mouse(left_pinch, right_pinch, now):
    global left_down, left_start, last_click_time, right_latched
    if left_pinch:
        if not left_down and now - last_click_time > CLICK_COOLDOWN:
            pyautogui.mouseDown()
            left_down = True
            left_start = now
    elif left_down:
        pyautogui.mouseUp()
        left_down = False
        last_click_time = now

    if right_pinch and not left_down and not right_latched:
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.rightClick()
            right_latched = True
            last_click_time = now
    elif not right_pinch:
        right_latched = False


def reset_tracking():
    global calibrating, thumb_history, initial_thumb_pos, smoothed
    global left_state, right_state, left_count, right_count, right_latched
    calibrating = True
    thumb_history = []
    initial_thumb_pos = None
    smoothed = None
    left_state = right_state = False
    left_count = right_count = 0
    right_latched = False
    release_mouse()


capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))
start_time = time.time()
last_ts = -1


def draw_hand(img, pts):
    h, w = img.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, xy[a], xy[b], (255, 255, 255), 1)
    for pt in xy:
        cv2.circle(img, pt, 2, (0, 0, 255), -1)

cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

try:
    while processing_active.is_set():
        if not new_frame_ready.wait(timeout=0.05):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        with frame_lock:
            if current_frame is None:
                new_frame_ready.clear()
                continue
            frame = current_frame.copy()
        new_frame_ready.clear()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()
        ts = max(int((now - start_time) * 1000), last_ts + 1)
        last_ts = ts
        results = hands.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        status = "NO HAND"

        if results.hand_landmarks:
            pts = results.hand_landmarks[0]
            thumb, index, middle, ring = pts[4], pts[8], pts[12], pts[16]

            thumb_history.append((thumb.x, thumb.y))
            if len(thumb_history) > 5:
                thumb_history.pop(0)
            avg_x = sum(p[0] for p in thumb_history) / len(thumb_history)
            avg_y = sum(p[1] for p in thumb_history) / len(thumb_history)

            if calibrating:
                status = "CALIBRATING"
                if len(thumb_history) >= CALIBRATION_FRAMES:
                    initial_thumb_pos = (avg_x, avg_y)
                    smoothed = (avg_x, avg_y)
                    pyautogui.moveTo(screen_width // 2, screen_height // 2)
                    calibrating = False
            else:
                move_cursor(avg_x, avg_y)

                left_state, left_count = debounce(dist(index, middle) < PINCH_THRESHOLD, left_state, left_count)
                right_state, right_count = debounce(dist(middle, ring) < PINCH_THRESHOLD, right_state, right_count)
                handle_mouse(left_state, right_state, now)

                if left_down:
                    held = now - left_start
                    status = f"HOLD {held:.1f}s" if held > HOLD_THRESHOLD else "CLICK"
                elif right_latched:
                    status = "RIGHT"
                else:
                    status = "MOVE"

            draw_hand(frame, pts)
        else:
            reset_tracking()

        view = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.putText(view, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_tracking()

except KeyboardInterrupt:
    print("Stopping program...")
finally:
    processing_active.clear()
    release_mouse()
    hands.close()
    cv2.destroyAllWindows()
    if capture_thread.is_alive():
        capture_thread.join(timeout=1.0)