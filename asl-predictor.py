import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# ================== CONFIG ==================
HAS_RESCALE = True                
IMG_SIZE = 128                    
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K',
    'L','M','N','O','P','Q','R','S','T','U','V',
    'W','X','Y','Z','del','nothing','space'
]

# Camera & preprocessing
FLIP_FRAME = True                 # mirror the webcam
TARGET_WIDTH = 1280               # ask cam for 1280x720
TARGET_HEIGHT = 720
USE_MJPG = True                   # request MJPG for better fps/detail 
PAD = 20                          # padding around detected hand
MIN_SIDE = 200                    # minimum square crop (keeps hand large enough)
USE_CLAHE = True                  # local contrast boost (try toggling)
USE_ROTATION_NORM = False          # rotate ROI so finger points up (helps stability)

# Prediction smoothing / confirmation
SMOOTH_N = 7                      # moving average window (frames)
CONF_THRESHOLD = 0.70             # min confidence to show a label
HOLD_N = 7                   

# Debug
SHOW_ROI = True                   # show the ROI in a second window
SAVE_KEY = ord('s')               # press 's' to save current ROI to disk
QUIT_KEY = ord('q')               # press 'q' to quit
# ============================================


# ---------- model ----------
load_dotenv()
model_path = os.getenv("MODEL_PATH")
if not os.path.exists(model_path):
    raise FileNotFoundError("MODEL_PATH not set or file not found. Put MODEL_PATH in .env or edit script.")

print(f"Loading model: {model_path}")
model = load_model(model_path)
try:
    print("Model input shape:", model.inputs[0].shape)
except Exception:
    pass


# ---------- mediapipe ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60
)


# ---------- helpers ----------
def hand_bbox_from_landmarks(landmarks, w, h, pad=20, min_side=0):
    """Return padded, clamped *square* bbox (x1,y1,x2,y2) from hand landmarks."""
    xs = [int(lmk.x * w) for lmk in landmarks.landmark]
    ys = [int(lmk.y * h) for lmk in landmarks.landmark]
    x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + pad)

    
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh, min_side if min_side else 0)
    cx, cy = x1 + bw // 2, y1 + bh // 2
    x1, x2 = cx - side // 2, cx + side // 2
    y1, y2 = cy - side // 2, cy + side // 2

    # Clamp to frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def rotate_to_upright(rgb_roi, hand_lms_fullframe, x1, y1, x2, y2):
    """
    Rotate ROI so the vector wrist(0) -> middle finger MCP(9) points up.
    hand_lms_fullframe: landmarks in full frame coords; we compute angle from them.
    """
    try:
        p0 = hand_lms_fullframe.landmark[0]   # wrist
        p9 = hand_lms_fullframe.landmark[9]   # middle finger MCP
        dx = (p9.x - p0.x)
        dy = (p9.y - p0.y)
        angle = -np.degrees(np.arctan2(dy, dx)) + 90  # rotate so finger points upwards

        h, w = rgb_roi.shape[:2]
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rot = cv2.warpAffine(rgb_roi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rot
    except Exception:
        return rgb_roi  # fallback


def apply_clahe_rgb(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# moving average of softmax vectors
probs_buffer = deque(maxlen=SMOOTH_N)
def smoothed_prediction(pred_vec):
    probs_buffer.append(pred_vec)
    avg = np.mean(probs_buffer, axis=0)
    idx = int(np.argmax(avg))
    conf = float(avg[idx])
    return idx, conf


# ---------- video ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ask for better resolution & codec
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
if USE_MJPG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("Press 'q' to quit, 's' to save current ROI.")
prev_t = time.time()

last_idx = None
streak = 0
stable_idx = None
stable_conf = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]

    # Mediapipe -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Draw & predict
    label_text = "…"
    conf_text = ""

    if res.multi_hand_landmarks:
        hand_lms = res.multi_hand_landmarks[0]
        # visualize landmarks 
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # robust bbox
        x1, y1, x2, y2 = hand_bbox_from_landmarks(hand_lms, W, H, pad=PAD, min_side=MIN_SIDE)
        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size > 0:
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

            if USE_ROTATION_NORM:
                roi_rgb = rotate_to_upright(roi_rgb, hand_lms, x1, y1, x2, y2)

            if USE_CLAHE:
                roi_rgb = apply_clahe_rgb(roi_rgb)

            roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            # scale according to model
            if HAS_RESCALE:
                roi_input = roi_resized.astype(np.float32)                 
            else:
                roi_input = (roi_resized.astype(np.float32) / 255.0)       #

            roi_input = np.expand_dims(roi_input, axis=0)

            # predict + smooth + hold-to-confirm
            pred = model.predict(roi_input, verbose=0)[0]
            cls_idx, conf = smoothed_prediction(pred)

            if cls_idx == last_idx:
                streak += 1
            else:
                last_idx = cls_idx
                streak = 1

            if streak >= HOLD_N and conf >= CONF_THRESHOLD:
                stable_idx = cls_idx
                stable_conf = conf

            if stable_idx is not None:
                label_text = CLASS_NAMES[stable_idx]
                conf_text = f" ({stable_conf*100:.1f}%)"

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # optional ROI window & saver
            if SHOW_ROI:
                cv2.imshow("ROI", cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))
    else:
        # no hand, reset streak 
        streak = 0
        last_idx = None

    # FPS
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now

    # HUD
    cv2.putText(frame, f"{label_text}{conf_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    cv2.imshow("ASL Real-Time Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == QUIT_KEY:
        break
    if key == SAVE_KEY and res.multi_hand_landmarks and 'roi_resized' in locals():
        fn = f"debug_roi_{int(time.time())}.jpg"
        cv2.imwrite(fn, cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))
        print("Saved:", fn)

cap.release()
hands.close()
cv2.destroyAllWindows()
