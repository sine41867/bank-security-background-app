import face_recognition
import numpy as np
class FaceRecognizer:
    def __init__(self, known_faces, known_names, face_types):
        self.known_faces = known_faces
        self.known_names = known_names
        self.known_face_types = face_types

    def recognize_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = []
        
        for face_location in face_locations:
            try:
                face_encodings.append(face_recognition.face_encodings(frame, [face_location])[0])
            except Exception as e:
                print(f"Error encoding face: {e}")
                continue

        face_names = []
        face_types = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_names[best_match_index]
                face_type = self.known_face_types[best_match_index]
                print(face_type)

            face_names.append(name)
            face_types.append(face_type)
        
        return face_locations, face_names, face_types 

