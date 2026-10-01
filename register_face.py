import cv2
import pickle
import os
import time
import numpy as np
from insightface.app import FaceAnalysis

DB_FILE = "face_database.pkl"

# Initialize InsightFace with Intel DirectML
print("Initializing face registration engine...")
app = FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

# Load existing database
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"Loaded existing database with {len(known_faces)} registered individuals.")
else:
    known_faces = {}

name = input("Enter your name: ").strip()
if not name:
    print("Name cannot be empty!")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))

recording = False
start_time = None
RECORDING_DURATION = 10.0  # Seconds to record
captured_embeddings = []

print("\n--- INSTRUCTIONS ---")
print("1. Press 'SPACE' to start the 5-second continuous recording.")
print("2. Slowly turn your head (center, left, right, up, down) while recording.")
print("3. Press 'Q' to cancel.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    current_time = time.time()

    faces = app.get(frame)

    if recording:
        elapsed = current_time - start_time
        remaining = max(0.0, RECORDING_DURATION - elapsed)

        if len(faces) > 0:
            # Pick the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = largest_face.embedding

            # Only save embedding if it captures a new/distinct angle (distance > 0.15)
            is_distinct = True
            for existing_emb in captured_embeddings:
                if cosine_distance(existing_emb, emb) < 0.15:
                    is_distinct = False
                    break

            if is_distinct:
                captured_embeddings.append(emb)

            box = largest_face.bbox.astype(int)
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, f"RECORDING... Turn Head Slowly", (box[0], max(box[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # UI Overlay during recording
        cv2.putText(display_frame, f"Time Left: {remaining:.1f}s | Captured Poses: {len(captured_embeddings)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if elapsed >= RECORDING_DURATION:
            recording = False
            if len(captured_embeddings) > 0:
                if name not in known_faces:
                    known_faces[name] = {"embeddings": [], "total_time": 0.0}

                # Save up to 10 distinct embeddings for this person
                known_faces[name]["embeddings"].extend(captured_embeddings)
                known_faces[name]["embeddings"] = known_faces[name]["embeddings"][:10]

                with open(DB_FILE, "wb") as f:
                    pickle.dump(known_faces, f)

                print(f"\nSuccessfully registered {name} with {len(known_faces[name]['embeddings'])} pose variations!")
                break
            else:
                print("\nNo valid face poses captured. Try again.")

    else:
        # UI Overlay before recording starts
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        cv2.putText(display_frame, "Press SPACE to start recording", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Multi-Angle Face Registration", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and not recording:
        recording = True
        start_time = time.time()
        captured_embeddings = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()