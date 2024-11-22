import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Directory containing images of registered individuals
KNOWN_FACES_DIR = 'known_faces'

# Initialize lists
known_face_encodings = []
known_face_names = []

# Load and encode known faces
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Initialize attendance dataframe
attendance_file = 'attendance.csv'
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(attendance_file, index=False)
else:
    df = pd.read_csv(attendance_file)

# Initialize video capture (0 for webcam, replace with ESP32-CAM stream URL if applicable)
video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name != "Unknown":
                if name not in df['Name'].values:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.concat([df, pd.DataFrame([{'Name': name, 'Time': current_time}])], ignore_index=True)
                    df.to_csv(attendance_file, index=False)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('SmartAttend - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()