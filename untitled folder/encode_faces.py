    import os
    import face_recognition
    import json

    face_database_path = '/Users/sohanreddy/Desktop/PPE_detection_using_YOLOV8/face_recognition/face_database'
    face_db = {}

    def encode_faces():
        # Check if the directory exists
        if not os.path.exists(face_database_path):
            print(f"Directory {face_database_path} does not exist!")
            return

        for filename in os.listdir(face_database_path):
            filepath = os.path.join(face_database_path, filename)

            # Skip non-file entries and non-image files
            if not os.path.isfile(filepath) or filename.startswith('.') or not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            print(f"Encoding face from {filename}...")
            try:
                image = face_recognition.load_image_file(filepath)
                face_encoding = face_recognition.face_encodings(image)
                
                if face_encoding:
                    face_db[filename] = face_encoding[0]
                else:
                    print(f"No face found in {filename}. Skipping...")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Save the encoded faces to a JSON file
        with open('face_encoding.json', 'w') as f:
            json.dump(face_db, f)
        print("Face encoding completed and saved to face_encoding.json")
