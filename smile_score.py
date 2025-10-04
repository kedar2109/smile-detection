import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ------------------ Mediapipe setup ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ Global variables ------------------
calibration_samples = []
is_calibrated = False
baseline_lip_gap = None
baseline_mouth_width = None
baseline_mouth_ratio = None
frame_count = 0
calibration_frames = 90  # neutral face ~3 sec

# Max smile calibration
max_lip_gap = None
max_mouth_width = None
max_mouth_ratio = None
smile_calibration_done = False
smile_calibration_frames = 90  # max smile ~3 sec

# Smoothing
score_history = deque(maxlen=5)
show_debug = False

# ------------------ Helper functions ------------------
def get_smile_features(landmarks, w, h):
    """Extract smile features: lip gap, mouth width, ratio"""
    upper_outer = np.array([landmarks[12].x * w, landmarks[12].y * h])
    lower_outer = np.array([landmarks[13].x * w, landmarks[13].y * h])
    upper_inner = np.array([landmarks[11].x * w, landmarks[11].y * h])
    lower_inner = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left_corner = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right_corner = np.array([landmarks[291].x * w, landmarks[291].y * h])

    lip_gap = np.linalg.norm(lower_inner - upper_inner)
    outer_gap = np.linalg.norm(lower_outer - upper_outer)
    mouth_width = np.linalg.norm(right_corner - left_corner)
    mouth_ratio = mouth_width / (outer_gap + 1e-6)

    return lip_gap, outer_gap, mouth_width, mouth_ratio

def detect_teeth_region(frame, landmarks, w, h):
    try:
        upper_lip = np.array([landmarks[11].x * w, landmarks[11].y * h], dtype=np.int32)
        lower_lip = np.array([landmarks[14].x * w, landmarks[14].y * h], dtype=np.int32)
        left_corner = np.array([landmarks[61].x * w, landmarks[61].y * h], dtype=np.int32)
        right_corner = np.array([landmarks[291].x * w, landmarks[291].y * h], dtype=np.int32)

        center_x = int((left_corner[0] + right_corner[0]) / 2)
        center_y = int((upper_lip[1] + lower_lip[1]) / 2)

        roi_size = 20
        x1 = max(0, center_x - roi_size)
        x2 = min(w, center_x + roi_size)
        y1 = max(0, center_y - roi_size)
        y2 = min(h, center_y + roi_size)

        if x2 > x1 and y2 > y1:
            mouth_roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_roi)
            return brightness
    except:
        pass
    return 0

def calculate_smile_score(lip_gap, outer_gap, mouth_width, mouth_ratio, teeth_brightness):
    global is_calibrated, baseline_lip_gap, baseline_mouth_width, baseline_mouth_ratio
    global max_lip_gap, max_mouth_width, max_mouth_ratio, smile_calibration_done

    if is_calibrated and smile_calibration_done:
        # Scale features dynamically between neutral and max smile
        gap_score = np.interp(lip_gap, [baseline_lip_gap, max_lip_gap], [0, 38])
        width_score = np.interp(mouth_width, [baseline_mouth_width, max_mouth_width], [0, 38])
        ratio_score = np.interp(mouth_ratio, [baseline_mouth_ratio, max_mouth_ratio], [0, 24])
        teeth_score = np.interp(teeth_brightness, [90, 170], [0, 10])


        teeth_score = 0
        if teeth_brightness > 90:
            teeth_score = np.interp(teeth_brightness, [90, 170], [0, 10])

        total_score = gap_score + width_score + ratio_score + teeth_score
    else:
        # Before calibration finishes
        total_score = np.interp(lip_gap, [5, 20], [0, 100])

    return int(np.clip(total_score, 0, 100))

# ------------------ Webcam capture ------------------
cap = cv2.VideoCapture(0)

print("=== Teeth & Lip Based Smile Detector ===")
print("Step 1: Neutral face calibration (~3 sec)")
print("Step 2: Max smile calibration (~3 sec)")
print("Press 'q' to quit, 'r' to recalibrate, 'd' for debug info\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lip_gap, outer_gap, mouth_width, mouth_ratio = get_smile_features(face_landmarks.landmark, w, h)
        teeth_brightness = detect_teeth_region(frame, face_landmarks.landmark, w, h)

        # ------------------ Neutral Calibration ------------------
        
        if not is_calibrated and frame_count < calibration_frames:
            calibration_samples.append((lip_gap, mouth_width, mouth_ratio))
            frame_count += 1

            progress = int((frame_count / calibration_frames) * 100)
            cv2.putText(frame, f"NEUTRAL FACE CALIBRATION {progress}%", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            if frame_count >= calibration_frames:
                gaps = [s[0] for s in calibration_samples]
                widths = [s[1] for s in calibration_samples]
                ratios = [s[2] for s in calibration_samples]

                baseline_lip_gap = np.median(gaps)
                baseline_mouth_width = np.median(widths)
                baseline_mouth_ratio = np.median(ratios)

                is_calibrated = True
                calibration_samples = []
                frame_count = 0
                print(f"✓ Neutral calibration done!")

        # ------------------ Max Smile Calibration ------------------
        elif is_calibrated and not smile_calibration_done and frame_count < smile_calibration_frames:
            calibration_samples.append((lip_gap, mouth_width, mouth_ratio))
            frame_count += 1

            cv2.putText(frame, f"SMILE CALIBRATION ({int(frame_count/smile_calibration_frames*100)}%)", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            if frame_count >= smile_calibration_frames:
                max_lip_gap = np.max([s[0] for s in calibration_samples])
                max_mouth_width = np.max([s[1] for s in calibration_samples])
                max_mouth_ratio = np.max([s[2] for s in calibration_samples])
                smile_calibration_done = True
                calibration_samples = []
                frame_count = 0
                print(f"✓ Smile calibration done!")

        # ------------------ Smile Detection ------------------
        elif is_calibrated and smile_calibration_done:
            score = calculate_smile_score(lip_gap, outer_gap, mouth_width, mouth_ratio, teeth_brightness)
            score_history.append(score)
            smoothed_score = int(np.mean(score_history))

            # Color & status
            if smoothed_score < 20:
                score_color = (255, 100, 100)
                status = "Neutral"
            elif smoothed_score < 50:
                score_color = (100, 200, 255)
                status = "Slight Smile"
            elif smoothed_score < 75:
                score_color = (100, 255, 200)
                status = "Smile"
            else:
                score_color = (100, 255, 100)
                status = "Big Smile!"

            # Display
            cv2.putText(frame, f"Smile: {smoothed_score}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, score_color, 4)
            cv2.putText(frame, status, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

            if show_debug:
                cv2.putText(frame, f"Lip Gap: {lip_gap:.1f} ({baseline_lip_gap:.1f}-{max_lip_gap:.1f})", 
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Mouth Width: {mouth_width:.1f} ({baseline_mouth_width:.1f}-{max_mouth_width:.1f})", 
                            (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.2f} ({baseline_mouth_ratio:.2f}-{max_mouth_ratio:.2f})", 
                            (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Teeth Brightness: {teeth_brightness:.1f}", (30, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        cv2.putText(frame, "No face detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smile Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
    elif key == ord('r'):
        # Reset everything
        calibration_samples = []
        is_calibrated = False
        smile_calibration_done = False
        frame_count = 0
        score_history.clear()
        print("\n=== Recalibrating ===")

cap.release()
cv2.destroyAllWindows()
