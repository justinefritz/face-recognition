import cv2
import time
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

print("Loading YOLO for Intel GPU...")
base_model = YOLO('yolov8n-face.pt')
base_model.export(format='openvino', dynamic=True, verbose=False)
model = YOLO('yolov8n-face_openvino_model/')

print("Loading InsightFace on Intel DirectML...")
app = FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Memory Database
# known_faces = { global_id: {"embeddings": [np.array, ...], "total_time": float} }
known_faces = {}    
active_tracks = {}  # { track_id: global_id }
track_frame_counts = {}  # Tracks how long a new track_id has been visible

COSINE_THRESHOLD = 0.72
MAX_GALLERY_SIZE = 5  # Store up to 5 face poses per person
next_person_id = 1

cached_faces = []

def cosine_distance(source, target):
    s = source / np.linalg.norm(source)
    t = target / np.linalg.norm(target)
    return float(1.0 - np.dot(s, t))

def min_gallery_distance(gallery, target_emb):
    """Finds the lowest cosine distance across all stored pose variations of a person."""
    return min([cosine_distance(saved_emb, target_emb) for saved_emb in gallery])

def get_center_distance(boxA, boxB):
    cxA, cyA = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    cxB, cyB = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    return np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

last_frame_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time
    frame_count += 1

    frame = cv2.flip(frame, 1)

    # 1. Periodically run InsightFace face analysis
    if frame_count % 8 == 0 or len(cached_faces) == 0:
        cached_faces = app.get(frame)

    # 2. Track heads using YOLO
    results = model.track(frame, imgsz=320, conf=0.45, persist=True, verbose=False)[0]
    present_count = 0

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        present_count = len(track_ids)

        # Cleanup lost tracks
        active_tracks = {tid: gid for tid, gid in active_tracks.items() if tid in track_ids}
        track_frame_counts = {tid: cnt for tid, cnt in track_frame_counts.items() if tid in track_ids}

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Increment visible frame counter for this track ID
            track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
            
            global_id = active_tracks.get(track_id)

            # Wait until track is stabilized (visible for >= 3 frames) before running embedding check
            if global_id is None and track_frame_counts[track_id] >= 3 and len(cached_faces) > 0:
                best_spatial_dist = float('inf')
                closest_face = None

                for face in cached_faces:
                    dist = get_center_distance((x1, y1, x2, y2), face.bbox)
                    if dist < best_spatial_dist and dist < 120:
                        best_spatial_dist = dist
                        closest_face = face

                if closest_face is not None:
                    current_emb = closest_face.embedding

                    best_match_id = None
                    min_dist = float('inf')

                    # Check against all stored pose variations for each known person
                    for f_id, f_data in known_faces.items():
                        d = min_gallery_distance(f_data["embeddings"], current_emb)
                        if d < min_dist:
                            min_dist = d
                            best_match_id = f_id

                    # Match found below threshold
                    if min_dist < COSINE_THRESHOLD:
                        global_id = best_match_id
                        # Add new pose variation to person gallery if space allows
                        if len(known_faces[global_id]["embeddings"]) < MAX_GALLERY_SIZE:
                            known_faces[global_id]["embeddings"].append(current_emb)
                    else:
                        # Register as new person
                        global_id = f"Person #{next_person_id}"
                        next_person_id += 1
                        known_faces[global_id] = {
                            "embeddings": [current_emb],
                            "total_time": 0.0
                        }

                    # Lock ID to the current active track
                    active_tracks[track_id] = global_id

            # Accumulate time duration
            if global_id and global_id in known_faces:
                known_faces[global_id]["total_time"] += dt
                accumulated_seconds = int(known_faces[global_id]["total_time"])
                label = f"{global_id} | Time: {accumulated_seconds}s"
                color = (0, 255, 0)
            else:
                label = f"Head #{track_id} (Stabilizing...)"
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Heads Present: {present_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Intel GPU Presence Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()