import cv2
import face_recognition
import json
import numpy as np
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO
import os
import time

# Load YOLO model
model = YOLO('yolov8s_custom.pt')

# Paths
encodings_file = 'face_recognition/face_encodings.json'
output_directory = 'non_helmet_images'
os.makedirs(output_directory, exist_ok=True)

# Load known faces
if os.path.exists(encodings_file):
    with open(encodings_file, 'r') as f:
        known_faces = json.load(f)
else:
    known_faces = {}

# Function to send email alerts
def send_alert_email(worker_id):
    sender_email = "worker.safety.sit@gmail.com"
    sender_password = "dxtipdpwejwbzocu"
    receiver_email = "manager.safety.sit@gmail.com"

    msg = EmailMessage()
    msg.set_content(f"Alert! Worker ID {worker_id} is not wearing a helmet.")
    msg['Subject'] = "Safety Alert"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print(f"Alert sent to {receiver_email} for Worker ID {worker_id}.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to save images of workers not wearing helmets
def save_non_helmet_image(frame, worker_id):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{worker_id}_{timestamp}.jpg"
    filepath = os.path.join(output_directory, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved image of Worker ID {worker_id} not wearing helmet at {filepath}")

# Process video or webcam feed
def process_video(source=0):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Detect safety equipment using YOLO
        results = model(frame, verbose=False)
        detected_classes = [model.names[int(c.cls)] for r in results for c in r.boxes]
        helmet_boxes = [r.boxes.xyxy[0] for r in results for c in r.boxes if model.names[int(c.cls)] == 'Helmet']

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Match face with known faces
            matches = face_recognition.compare_faces(
                [np.array(v) for v in known_faces.values()], face_encoding
            )
            distances = face_recognition.face_distance(
                [np.array(v) for v in known_faces.values()], face_encoding
            )
            best_match_index = np.argmin(distances) if distances.size > 0 else -1

            if best_match_index != -1 and matches[best_match_index]:
                worker_id = list(known_faces.keys())[best_match_index]
            else:
                worker_id = "Unknown"

            # Check if the worker is wearing a helmet
            wearing_helmet = False
            for helmet_box in helmet_boxes:
                x1, y1, x2, y2 = map(int, helmet_box)
                # Check if the helmet overlaps with the face bounding box
                top, right, bottom, left = face_location
                if x1 < right and x2 > left and y1 < bottom and y2 > top:
                    wearing_helmet = True
                    break

            # Draw bounding box and label
            top, right, bottom, left = face_location
            color = (0, 255, 0) if wearing_helmet else (0, 0, 255)  # Green for Helmet, Red for No Helmet
            label = "Helmet" if wearing_helmet else "No Helmet"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{label} - ID: {worker_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save image and send alert if not wearing a helmet
            if not wearing_helmet and worker_id != "Unknown":
                send_alert_email(worker_id)
                save_non_helmet_image(frame, worker_id)

        # Draw bounding boxes around helmets
        for helmet_box in helmet_boxes:
            x1, y1, x2, y2 = map(int, helmet_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for helmets

        # Display the frame
        cv2.imshow('Safety Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    print("Choose video source:")
    print("1. Webcam")
    print("2. Upload video file")
    choice = int(input("Enter your choice (1/2): "))

    if choice == 1:
        process_video(0)  # Webcam feed
    elif choice == 2:
        video_path = input("Enter the path of the video file: ")
        if os.path.exists(video_path):
            process_video(video_path)
        else:
            print("Invalid video file path.")