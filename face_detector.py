import cv2
import time
import os
import csv
import pickle
import gc
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

DB_FILE = "face_database.pkl"
CSV_FILE = "presence_log.csv"

# Initialize CSV file with headers if missing
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Session Duration (s)"])

def log_event(name, action, duration="-"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, action, duration])

# Lightweight detector setup
app = FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(128, 128))  # Reduced detection resolution for speed

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# Capture at lower resolution to save CPU/RAM bandwidth directly at hardware level
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

person_sessions = {}      # { global_id: {"start_time": float, "last_seen": float} }

COSINE_THRESHOLD = 0.72
SIT_OUT_TIMEOUT = 2.0     # Seconds of absence before SIT_OUT
FACE_SKIP_FRAMES = 5      # Run heavy AI analysis only every Nth frame
next_person_id = 1

def cosine_distance(source, target):
    s = source / np.linalg.norm(source)
    t = target / np.linalg.norm(target)
    return float(1.0 - np.dot(s, t))

def min_gallery_distance(gallery, target_emb):
    return min([cosine_distance(saved_emb, target_emb) for saved_emb in gallery])

frame_count = 0
cached_detected_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    frame = cv2.flip(frame, 1)

    # 1. LIGHTWEIGHT INFERENCE: Run AI detection only every Nth frame
    if frame_count % FACE_SKIP_FRAMES == 0 or len(cached_detected_faces) == 0:
        # Resize frame before AI model processing to minimize CPU math
        small_frame = cv2.resize(frame, (240, 180))
        scale_x = frame.shape[1] / 240.0
        scale_y = frame.shape[0] / 180.0

        raw_faces = app.get(small_frame)
        
        cached_detected_faces = []
        for face in raw_faces:
            # Rescale coordinates to display size
            face.bbox[0] *= scale_x
            face.bbox[2] *= scale_x
            face.bbox[1] *= scale_y
            face.bbox[3] *= scale_y
            cached_detected_faces.append(face)

        # Force garbage collector to release intermediate NumPy buffers
        if frame_count % 100 == 0:
            gc.collect()

    currently_visible_persons = set()

    # 2. MATCH & DISPLAY DETECTED FACES
    for face in cached_detected_faces:
        box = face.bbox.astype(int)
        current_emb = face.embedding

        best_match_id = None
        min_dist = float('inf')

        # Match embedding against gallery
        for f_id, f_data in known_faces.items():
            d = min_gallery_distance(f_data["embeddings"], current_emb)
            if d < min_dist:
                min_dist = d
                best_match_id = f_id

        if min_dist < COSINE_THRESHOLD:
            global_id = best_match_id
        else:
            global_id = f"Visitor #{next_person_id}"
            next_person_id += 1
            known_faces[global_id] = {
                "embeddings": [current_emb],
                "total_time": 0.0
            }

        currently_visible_persons.add(global_id)

        # LOG SIT_IN EVENT
        if global_id not in person_sessions:
            person_sessions[global_id] = {
                "start_time": current_time,
                "last_seen": current_time
            }
            log_event(global_id, "SIT_IN")
        else:
            person_sessions[global_id]["last_seen"] = current_time

        # UI Rendering
        color = (0, 255, 0) if not global_id.startswith("Visitor") else (0, 255, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, global_id, (box[0], max(box[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. CHECK FOR SIT_OUT EVENTS
    finished_sessions = []
    for g_id, session_data in person_sessions.items():
        if g_id not in currently_visible_persons:
            if current_time - session_data["last_seen"] > SIT_OUT_TIMEOUT:
                duration = int(session_data["last_seen"] - session_data["start_time"])
                log_event(g_id, "SIT_OUT", duration=f"{duration}s")
                finished_sessions.append(g_id)

    for g_id in finished_sessions:
        del person_sessions[g_id]

    cv2.imshow("Face Recognition", frame)

    # 4. CAP FRAME SPEED to prevent CPU spin lock
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Handle SIT_OUT on close
for g_id, session_data in person_sessions.items():
    duration = int(time.time() - session_data["start_time"])
    log_event(g_id, "SIT_OUT (App Closed)", duration=f"{duration}s")

cap.release()
cv2.destroyAllWindows()

with open(DB_FILE, "wb") as f:
    pickle.dump(known_faces, f)