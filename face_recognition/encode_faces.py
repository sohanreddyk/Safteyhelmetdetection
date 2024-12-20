import os
import face_recognition
import json

# Paths
face_database_path = 'face_recognition/face_database'
encodings_file = 'face_recognition/face_encodings.json'

# Dictionary to store face encodings
face_db = {}

def encode_faces():
    # Ensure the database path exists
    if not os.path.exists(face_database_path):
        print(f"Directory {face_database_path} does not exist!")
        return

    for filename in os.listdir(face_database_path):
        filepath = os.path.join(face_database_path, filename)

        # Skip .DS_Store and non-image files
        if filename == '.DS_Store' or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        print(f"Encoding face from {filename}...")
        try:
            # Load the image and get face encodings
            image = face_recognition.load_image_file(filepath)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                # Use the filename without extension as the worker ID
                worker_id = os.path.splitext(filename)[0]
                face_db[worker_id] = face_encodings[0].tolist()  # Convert to list for JSON serialization
            else:
                print(f"No face found in {filename}. Skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the encoded faces to a JSON file
    with open(encodings_file, 'w') as f:
        json.dump(face_db, f)
    print(f"Face encoding completed and saved to {encodings_file}")

if __name__ == "__main__":
    encode_faces()
