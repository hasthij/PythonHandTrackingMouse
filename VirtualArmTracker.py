import sys
import subprocess

# List of required packages and their pip names
required_packages = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pyautogui": "pyautogui",
    "numpy": "numpy"
}

# Install any missing packages
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
import numpy as np
import math

# Extreme Performance Configuration
FRAME_WIDTH = 160  # Reduced for faster processing
FRAME_HEIGHT = 120
PINCH_THRESHOLD = 0.046  # Optimized sensitivity for click detection
MOUSE_SENSITIVITY = 15.0  # Increased responsiveness
POSITION_SMOOTHING = 0.8  # More responsive movement
DEADZONE = 0.2  # Further reduced deadzone
MODEL_COMPLEXITY = 0  # Fastest model
MAX_QUEUE_SIZE = 2  # Limit frame queue size

# Hold parameters - improved for stability
HOLD_THRESHOLD = 0.35  # Time in seconds before registering a hold
HOLD_STABILITY_FRAMES = 6  # Number of consistent frames needed before changing state

CLICK_COOLDOWN = 0.2  # Time in seconds between clicks to prevent jitter

# Zero delay settings
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Initialize MediaPipe with minimal settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# State variables
hand_landmarks = None
initial_thumb_pos = None
calibration_center = None
left_click_active = False
right_click_detected = False
hold_mode_active = False
thumb_pos_history = []
smoothed_x, smoothed_y = None, None
last_x, last_y = None, None
last_left_click_state = False
last_right_click_state = False
click_hold_start_time = None
last_click_time = 0
capture_initial_pos = True
left_pinch_confidence = 0
right_pinch_confidence = 0
hold_confidence = 0
last_frame_time = time.time()
fps_values = []
mouse_is_down = False

# Thread synchronization
processing_active = threading.Event()
processing_active.set()
frame_lock = threading.Lock()
new_frame_ready = threading.Event()

# Frame buffer (minimized)
current_frame = None
processing_frame = None

# Streamlined distance calculation
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# Optimized frame capture thread
def capture_frames():
    global current_frame, new_frame_ready
    
    # Initialize webcam with optimized settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Direct Show API for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
    cap.set(cv2.CAP_PROP_FPS, 120)  # Request maximum FPS
    
    # Try to use MJPG for faster processing
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except:
        pass
    
    # Set thread priority if possible
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except:
        pass
    
    while processing_active.is_set():
        ret, frame = cap.read()
        if ret:
            # Flip and resize in a single operation
            frame = cv2.flip(frame, 1)
            
            with frame_lock:
                current_frame = frame
                new_frame_ready.set()
        else:
            # Minimal sleep to prevent CPU overload on failure
            time.sleep(0.001)
    
    # Release camera when done
    cap.release()

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Initialize hand detection with minimal settings
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=MODEL_COMPLEXITY)

# Streamlined drawing settings
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Optimized thumb-based cursor movement
def update_cursor_position(thumb_x, thumb_y):
    global last_x, last_y, smoothed_x, smoothed_y, initial_thumb_pos
    
    # First-time initialization
    if last_x is None:
        last_x, last_y = thumb_x, thumb_y
        smoothed_x, smoothed_y = thumb_x, thumb_y
        return
    
    # Apply direct smoothing
    smoothed_x = smoothed_x + POSITION_SMOOTHING * (thumb_x - smoothed_x)
    smoothed_y = smoothed_y + POSITION_SMOOTHING * (thumb_y - smoothed_y)
    
    # Map to screen with optimized calculation
    if initial_thumb_pos:
        # Fast relative position calculation
        rel_x = (smoothed_x - initial_thumb_pos[0]) / (FRAME_WIDTH * 0.4)
        rel_y = (smoothed_y - initial_thumb_pos[1]) / (FRAME_HEIGHT * 0.4)
        
        # Direct screen mapping
        screen_x = min(max(0, screen_width/2 + rel_x * screen_width * 0.7), screen_width)
        screen_y = min(max(0, screen_height/2 + rel_y * screen_height * 0.7), screen_height)
        
        # Move cursor without any additional checks
        pyautogui.moveTo(int(screen_x), int(screen_y))
    
    last_x, last_y = smoothed_x, smoothed_y

# Improved mouse action management
def handle_mouse_actions(is_left_pinching, is_right_pinching):
    global left_click_active, right_click_detected, hold_mode_active
    global click_hold_start_time, mouse_is_down, last_click_time, hold_confidence
    
    current_time = time.time()
    
    # Manage left click and hold with stability checks
    if is_left_pinching:
        # Pinch detected - start tracking for potential hold
        if not left_click_active:
            # Check if we're not in cooldown period
            if current_time - last_click_time > CLICK_COOLDOWN:
                # New pinch detected - start tracking
                if not mouse_is_down:
                    pyautogui.mouseDown()
                    mouse_is_down = True
                    click_hold_start_time = current_time
                left_click_active = True
                hold_mode_active = False
                hold_confidence = 0
        else:
            # Already in left-click: check if we should transition to hold mode
            if click_hold_start_time and not hold_mode_active:
                hold_duration = current_time - click_hold_start_time
                if hold_duration > HOLD_THRESHOLD:
                    # Increase confidence for hold mode
                    hold_confidence += 1
                    if hold_confidence >= HOLD_STABILITY_FRAMES:
                        hold_mode_active = True
    else:
        # No pinch - release if needed
        if left_click_active:
            # Reset hold tracking
            hold_mode_active = False
            hold_confidence = 0
            left_click_active = False
            
            # Only release if we were holding
            if mouse_is_down:
                pyautogui.mouseUp()
                mouse_is_down = False
                last_click_time = current_time
    
    # Handle right click (simpler, no hold needed)
    if is_right_pinching and not left_click_active and not right_click_detected:
        if current_time - last_click_time > CLICK_COOLDOWN:
            pyautogui.rightClick()
            right_click_detected = True
            last_click_time = current_time
    elif not is_right_pinching:
        right_click_detected = False

# Main loop with optimized processing
try:
    while processing_active.is_set():
        # Wait for new frame with timeout
        if not new_frame_ready.wait(timeout=0.016):  # ~60fps timeout
            continue
        
        # Get frame with minimal locking
        with frame_lock:
            if current_frame is None:
                new_frame_ready.clear()
                continue
            
            processing_frame = current_frame.copy()
            new_frame_ready.clear()
        
        # Process frame with minimal operations
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Skip unnecessary copying
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Calculate FPS occasionally
        current_time = time.time()
        if len(fps_values) % 10 == 0:
            delta_time = current_time - last_frame_time
            if delta_time > 0:
                fps = 1 / delta_time
                fps_values.append(fps)
                if len(fps_values) > 10:
                    fps_values.pop(0)
        last_frame_time = current_time
        
        # Handle hand landmarks if detected
        hand_visible = False
        if results.multi_hand_landmarks:
            hand_visible = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract only the landmarks we need
            index_tip = hand_landmarks.landmark[8]   # Index fingertip
            middle_tip = hand_landmarks.landmark[12]  # Middle fingertip
            ring_tip = hand_landmarks.landmark[16]   # Ring fingertip
            thumb_tip = hand_landmarks.landmark[4]   # Thumb tip
            
            # Convert thumb to pixel coordinates for cursor control
            thumb_x = int(thumb_tip.x * FRAME_WIDTH)
            thumb_y = int(thumb_tip.y * FRAME_HEIGHT)
            
            # Add to history for stabilization
            thumb_pos_history.append((thumb_x, thumb_y))
            if len(thumb_pos_history) > 5:  # Reduced history size for faster response
                thumb_pos_history.pop(0)
            
            # Fast averaging for stability
            stable_thumb_x = sum(p[0] for p in thumb_pos_history) / len(thumb_pos_history)
            stable_thumb_y = sum(p[1] for p in thumb_pos_history) / len(thumb_pos_history)
            
            # Direct gesture detection - distances between fingertips
            index_middle_distance = calculate_distance(index_tip, middle_tip)  # For left click
            middle_ring_distance = calculate_distance(middle_tip, ring_tip)    # For right click
            
            # Handle initial position capture
            if capture_initial_pos:
                if len(thumb_pos_history) >= 5:
                    initial_thumb_pos = (stable_thumb_x, stable_thumb_y)
                    pyautogui.moveTo(screen_width // 2, screen_height // 2)
                    smoothed_x, smoothed_y = stable_thumb_x, stable_thumb_y
                    last_x, last_y = stable_thumb_x, stable_thumb_y
                    capture_initial_pos = False
            else:
                # Update cursor position (always update during movement)
                update_cursor_position(stable_thumb_x, stable_thumb_y)
                
                # Detect gestures with stability thresholds
                is_left_pinching = index_middle_distance < PINCH_THRESHOLD
                is_right_pinching = middle_ring_distance < PINCH_THRESHOLD
                
                # State confidence tracking
                if is_left_pinching == last_left_click_state:
                    left_pinch_confidence = min(10, left_pinch_confidence + 1)
                else:
                    left_pinch_confidence = max(0, left_pinch_confidence - 1)
                
                if is_right_pinching == last_right_click_state:
                    right_pinch_confidence = min(10, right_pinch_confidence + 1)
                else:
                    right_pinch_confidence = max(0, right_pinch_confidence - 1)
                
                # Only act on stable states (confidence above threshold)
                stable_left_pinch = left_pinch_confidence >= 2
                stable_right_pinch = right_pinch_confidence >= 2
                
                # Handle mouse actions with improved stability
                handle_mouse_actions(
                    is_left_pinching=stable_left_pinch and is_left_pinching,
                    is_right_pinching=stable_right_pinch and is_right_pinching
                )
                
                # Remember current state for next frame
                last_left_click_state = is_left_pinching
                last_right_click_state = is_right_pinching
                
                # Determine current action for display
                action_text = "HOLD" if hold_mode_active else "CLICK" if left_click_active else "RIGHT" if right_click_detected else "MOVE"
                
                # Update window title with current status
                if hold_mode_active:
                    window_title = f"Hand Control - HOLD MODE ({int((current_time - click_hold_start_time)*10)/10}s)"
                else:
                    window_title = f"Hand Control - {action_text}"
                cv2.setWindowTitle('Zero Delay Hand Control', window_title)
            
            # Draw landmarks if resolution allows
            if FRAME_WIDTH > 120:
                mp_drawing.draw_landmarks(
                    processing_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_spec,
                    drawing_spec)
        
        # Handle case when hand disappears
        if not hand_visible:
            # Reset tracking state
            capture_initial_pos = True
            thumb_pos_history = []
            
            # Ensure mouse is released if it was down
            if mouse_is_down:
                pyautogui.mouseUp()
                mouse_is_down = False
            
            # Reset all click states
            left_click_active = False
            right_click_detected = False
            hold_mode_active = False
            hold_confidence = 0
            
            # Update window title
            cv2.setWindowTitle('Zero Delay Hand Control', "Hand Control - NO HAND")
        
        # Display frame
        cv2.imshow('Zero Delay Hand Control', processing_frame)
        
        # Check for quit key with minimal blocking
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Manual recalibration
            capture_initial_pos = True
            thumb_pos_history = []
            initial_thumb_pos = None

except KeyboardInterrupt:
    print("Stopping program...")
finally:
    # Clean shutdown
    processing_active.clear()
    
    # Release resources
    hands.close()
    cv2.destroyAllWindows()
    
    # Ensure mouse is released
    if mouse_is_down:
        pyautogui.mouseUp()
    
    # Wait for thread completion
    if capture_thread.is_alive():
        capture_thread.join(timeout=0.5) 