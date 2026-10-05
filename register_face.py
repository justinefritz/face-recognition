import cv2
import pickle
import os
import time
import gc
import numpy as np
from insightface.app import FaceAnalysis

DB_FILE = "face_database.pkl"

print("Initializing lightweight InsightFace registration engine...")
app = FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
# Reduced detection resolution for ultra-fast, low-memory inference
app.prepare(ctx_id=0, det_size=(128, 128))

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

name = input("Enter your name: ").strip()
if not name:
    print("Name cannot be empty!")
    exit()

# Set lower resolution directly at camera hardware layer
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))

recording = False
start_time = None
RECORDING_DURATION = 15.0
captured_embeddings = []

FRAME_SKIP = 5       # Process AI inference only every 5th frame
frame_count = 0
cached_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    current_time = time.time()

    # LIGHTWEIGHT INFERENCE: Run detection on skipped frames
    if frame_count % FRAME_SKIP == 0 or len(cached_faces) == 0:
        # Resize frame before model execution
        small_frame = cv2.resize(frame, (240, 180))
        scale_x = frame.shape[1] / 240.0
        scale_y = frame.shape[0] / 180.0
        
        raw_faces = app.get(small_frame)
        
        cached_faces = []
        for f in raw_faces:
            f.bbox[0] *= scale_x
            f.bbox[2] *= scale_x
            f.bbox[1] *= scale_y
            f.bbox[3] *= scale_y
            cached_faces.append(f)

        # Force Python garbage collection every 60 frames to keep RAM minimal
        if frame_count % 60 == 0:
            gc.collect()

    faces = cached_faces

    if recording:
        elapsed = current_time - start_time
        remaining = max(0.0, RECORDING_DURATION - elapsed)

        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = largest_face.embedding

            is_distinct = True
            for existing_emb in captured_embeddings:
                if cosine_distance(existing_emb, emb) < 0.15:
                    is_distinct = False
                    break

            if is_distinct:
                captured_embeddings.append(emb)

            box = largest_face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, "RECORDING... Turn Head Slowly", (box[0], max(box[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f"Time Left: {remaining:.1f}s | Poses: {len(captured_embeddings)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if elapsed >= RECORDING_DURATION:
            recording = False
            if len(captured_embeddings) > 0:
                if name not in known_faces:
                    known_faces[name] = {"embeddings": [], "total_time": 0.0}

                known_faces[name]["embeddings"].extend(captured_embeddings)
                known_faces[name]["embeddings"] = known_faces[name]["embeddings"][:15]

                with open(DB_FILE, "wb") as f:
                    pickle.dump(known_faces, f)

                print(f"\nSuccessfully registered {name} with {len(known_faces[name]['embeddings'])} pose variations!")
                break
            else:
                print("\nNo valid face poses captured. Try again.")

    else:
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        cv2.putText(frame, "Press SPACE to start recording", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Multi-Angle Face Registration", frame)

    # Frame delay to yield CPU back to the OS
    key = cv2.waitKey(20) & 0xFF
    if key == ord(' ') and not recording:
        recording = True
        start_time = time.time()
        captured_embeddings = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()