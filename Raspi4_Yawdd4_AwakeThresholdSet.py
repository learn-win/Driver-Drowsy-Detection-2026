import os
# Fix for Raspberry Pi Wayland / Font display errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import dlib
import numpy as np
from collections import deque
import onnxruntime as ort

# ==========================================
# 1. CONFIGURATION & SCALER
# ==========================================
MODEL_PATH = "vbfllfa_drowsiness.onnx"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
THRESHOLD = 0.6 
REQUIRED_FRAMES = 16 # Must be drowsy for 16 straight frames
AWAKE_EAR_THRESHOLD = 0.3 # If EAR is above this, instantly reset to Alert

# 🚨 REPLACE THESE WITH THE ACTUAL MEANS AND STDs FROM YOUR TRAINING NOTEBOOK
FEATURE_MEANS = np.array([0.3036, 0.1053, 70.6648, 0.2012], dtype=np.float32)
FEATURE_STDS  = np.array([0.0739, 0.1057, 160.3662, 0.2194], dtype=np.float32)

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("[INFO] Loading ONNX Runtime Session...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print(f"[ERROR] Could not load {PREDICTOR_PATH}. Place it in the script folder.")
    exit()

cap = cv2.VideoCapture(0)

# The model requires exactly 16 frames to make a prediction
feature_window = deque(maxlen=16)

# 150 frames (approx. 5 seconds of memory) to prevent sticky PERCLOS
ear_history = deque(maxlen=150) 

# Strict consecutive counter
consecutive_drowsy_frames = 0

# ==========================================
# 3. MATHEMATICAL FUNCTIONS
# ==========================================
def compute_ear(eye: np.ndarray) -> float:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth: np.ndarray) -> float:
    A = np.linalg.norm(mouth[2] - mouth[6])
    B = np.linalg.norm(mouth[3] - mouth[5])
    C = np.linalg.norm(mouth[0] - mouth[4])
    return (A + B) / (2.0 * C)

def compute_head_tilt(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    dx = float(right_center[0] - left_center[0])
    dy = float(right_center[1] - left_center[1])
    return np.degrees(np.arctan2(dy, dx))

def compute_perclos(ear_values: np.ndarray, threshold: float = 0.25) -> float:
    if len(ear_values) == 0: return 0.0
    closed_frames = int(np.sum(ear_values < threshold))
    return float(closed_frames / len(ear_values))

print("[INFO] Starting ONNX Transformer Drowsiness Detection (With Instant Reset Override)...")

# ==========================================
# 4. MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: 
        print("[ERROR] Camera feed lost.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = predictor(gray, face)
        
        # Convert landmarks to numpy array
        coords = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            
        # Extract features
        left_eye = coords[42:48]
        right_eye = coords[36:42]
        mouth = coords[60:68]

        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
        mar = compute_mar(mouth)
        tilt = compute_head_tilt(left_eye, right_eye)
        
        ear_history.append(ear)
        perclos = compute_perclos(np.array(ear_history))

        # Add to our 16-frame sequence
        feature_window.append([ear, mar, tilt, perclos])

        # Draw face box for UI
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)

        # 🚨 ONLY INFER IF WE HAVE A FULL 16-FRAME WINDOW
        if len(feature_window) == 16:
            # Convert to numpy and apply StandardScaler logic
            raw_data = np.array(feature_window, dtype=np.float32)
            scaled_data = (raw_data - FEATURE_MEANS) / FEATURE_STDS
            
            # Reshape to (1, 16, 4)
            input_tensor = np.expand_dims(scaled_data, axis=0)

            # Run ONNX Inference
            ort_inputs = {input_name: input_tensor}
            ort_outs = session.run(None, ort_inputs)
            
            # Get Logit Output and apply Sigmoid manually
            logit = ort_outs[0][0][0]
            probability = 1.0 / (1.0 + np.exp(-logit))

            # ==========================================
            # STRICT RESET & OVERRIDE LOGIC
            # ==========================================
            # If the raw EAR shows the eyes are clearly wide open right now...
            if ear >= AWAKE_EAR_THRESHOLD:
                consecutive_drowsy_frames = 0
                probability = 0.0 # Force UI to show 0% fatigue
                
            # Otherwise, rely on the Transformer's probability
            elif probability > THRESHOLD:
                consecutive_drowsy_frames += 1
                
            # If Transformer says awake, reset
            else:
                consecutive_drowsy_frames = 0
            
            # Trigger Drowsy Alarm ONLY if count reaches REQUIRED_FRAMES (16)
            if consecutive_drowsy_frames >= REQUIRED_FRAMES:
                final_status, status_color = "DROWSY - WAKE UP!", (0, 0, 255) # Red
                consecutive_drowsy_frames = REQUIRED_FRAMES # Cap it
            else:
                final_status, status_color = "ALERT", (0, 255, 0) # Green

            # Draw UI on Frame
            cv2.putText(frame, f"Status: {final_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"Fatigue Prob: {probability*100:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            votes_text = f"Drowsy Frames: {consecutive_drowsy_frames}/{REQUIRED_FRAMES}"
            cv2.putText(frame, votes_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    else:
        # Prevent the UI from freezing. Instantly reset count if face is lost.
        consecutive_drowsy_frames = 0
        cv2.putText(frame, "SEARCHING FOR FACE...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        votes_text = f"Drowsy Frames: 0/{REQUIRED_FRAMES}"
        cv2.putText(frame, votes_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    cv2.imshow("Raspberry Pi Drowsiness Transformer (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
