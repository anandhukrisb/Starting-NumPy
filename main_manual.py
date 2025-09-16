# This is the "True" Manual Code
import cv2
import numpy as np
import os
from PIL import Image

# --- Constants and Model Loading ---
KNOWN_FACES_DIR = 'known_faces'
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
embedder = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')
RECOGNITION_THRESHOLD = 0.6 
known_face_embeddings = []
known_face_names = []

# --- Step 1: Process Known Faces and Build the Database ---
print("Processing known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.1, 4)
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi = image[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        embedding = embedder.forward()
        known_face_embeddings.append(embedding.flatten())
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"Processed and stored embedding for: {os.path.splitext(filename)[0]}")

print("Database of known faces built.")

# --- Step 2: Real-time Recognition with Webcam ---
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if not ret: break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(gray_frame, 1.1, 4)
    for (x, y, w, h) in detected_faces:
        face_roi = frame[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        live_embedding = embedder.forward().flatten()
        distances = []
        for known_embedding in known_face_embeddings:
            dist = np.linalg.norm(known_embedding - live_embedding)
            distances.append(dist)
        name = "Unknown"
        if distances:
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            if min_distance < RECOGNITION_THRESHOLD:
                name = known_face_names[min_distance_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
video_capture.release()
cv2.destroyAllWindows()