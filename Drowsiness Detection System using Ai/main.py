import cv2
import time
import mediapipe as mp
from scipy.spatial import distance
import threading
import pygame

# --- Pygame Audio Setup (Low Latency) ---
pygame.mixer.pre_init(44100, -16, 2, 512)   # Lower buffer for minimal delay
pygame.init()
pygame.mixer.init()

SIREN_SOUND = "siren.mp3"
try:
    pygame.mixer.music.load(SIREN_SOUND)
except pygame.error as e:
    print(f"Could not load siren sound file: {e}")
    print("Please make sure 'siren.mp3' is in the same directory.")
    exit()

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0)
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [
    13, 14,
    78, 308,
    81, 178, 87,
    311, 402, 317
]

# --- Sound/Siren Thread Control ---
sound_lock = threading.Lock()
is_siren_playing = False

def play_siren_thread():
    global is_siren_playing
    with sound_lock:
        if not is_siren_playing:
            pygame.mixer.music.play(-1)
            is_siren_playing = True

def stop_siren_thread():
    global is_siren_playing
    with sound_lock:
        if is_siren_playing:
            pygame.mixer.music.stop()
            is_siren_playing = False

# --- Helper Functions ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return 0 if C == 0 else (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    if len(mouth_landmarks) < len(MOUTH_IDX): return 0
    A = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    C = distance.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    return 0 if C == 0 else A / C

# --- Tuning Thresholds and State Vars ---
EAR_THRESHOLD = 0.25
DROWSY_DURATION = 1.0
YAWN_THRESHOLD = 0.75
YAWN_DURATION = 1.0
start_time = None
yawn_start_time = None
enhance_mode = False

# --- Start Video Loop ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    drowsy_flag = False
    yawning_flag = False
    face_detected_flag = False

    # Colors for overlays
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    # --- Face and Landmark Detection ---
    if result.multi_face_landmarks:
        face_detected_flag = True
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        left_eye = []
        right_eye = []
        mouth = []

        for idx in LEFT_EYE_IDX:
            if 0 <= idx < len(landmarks):
                left_eye.append((int(landmarks[idx].x * w), int(landmarks[idx].y * h)))
        for idx in RIGHT_EYE_IDX:
            if 0 <= idx < len(landmarks):
                right_eye.append((int(landmarks[idx].x * w), int(landmarks[idx].y * h)))
        for idx in MOUTH_IDX:
            if 0 <= idx < len(landmarks):
                mouth.append((int(landmarks[idx].x * w), int(landmarks[idx].y * h)))

        # Draw landmarks
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, drawing_spec.circle_radius, drawing_spec.color, thickness=drawing_spec.thickness)

        # --- Compute Aspect Ratios and Display ---
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0 if len(left_eye)==6 and len(right_eye)==6 else 1.0
        mar = mouth_aspect_ratio(mouth) if len(mouth)==len(MOUTH_IDX) else 0.0

        cv2.putText(frame, f'EAR: {ear:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.putText(frame, f'MAR: {mar:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)

        # --- Drowsiness/Yawning Detection Logic ---
        if ear < EAR_THRESHOLD:
            if start_time is None: start_time = time.time()
            elif time.time() - start_time >= DROWSY_DURATION: drowsy_flag = True
        else:
            start_time = None
        if mar > YAWN_THRESHOLD:
            if yawn_start_time is None: yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= YAWN_DURATION: yawning_flag = True
        else:
            yawn_start_time = None

        # --- Display State ---
        status_shown = False
        if drowsy_flag:
            cv2.putText(frame, "DROWSY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
            status_shown = True
        if yawning_flag:
            cv2.putText(frame, "YAWNING", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 3)
            status_shown = True
        if not status_shown:
            cv2.putText(frame, "NORMAL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
    else:
        face_detected_flag = False
        start_time = None
        yawn_start_time = None
        cv2.putText(frame, "NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)

    # --- Enhancement Mode (Press 'L') ---
    if enhance_mode:
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        cv2.putText(frame, "ENHANCED", (430, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    # --- Siren Control: Immediate on Drowsy/Yawning/No Face ---
    if drowsy_flag or yawning_flag or (not face_detected_flag):
        play_siren_thread()
    else:
        stop_siren_thread()

    cv2.imshow('Drowsiness & Yawning Detection', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    if key in (ord('l'), ord('L')):
        enhance_mode = not enhance_mode
    pygame.event.pump()  # Keep mixer responsive

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
