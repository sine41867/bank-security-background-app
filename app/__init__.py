from app.models.alert_generator import AlertGenerator
from app.models.camera_handler import CameraHandler
from app.models.database_manager import DatabaseManager
from app.models.face_recognizer import FaceRecognizer
from app.models.alert import Alert
import cv2
import datetime

def app():
    
    db_manager = DatabaseManager()
    
    known_faces, known_names, face_types = db_manager.load_known_faces()
    
    #testing
    if not known_faces:
        print("No known faces loaded from the database.")
        return

    face_recognizer = FaceRecognizer(known_faces, known_names, face_types)
    
    camera = CameraHandler()

    alert_generator = AlertGenerator()

    while True:
        try:
            real_time_capture(camera, face_recognizer, db_manager, alert_generator)
        except Exception as e:
             print(str(e))
    

def real_time_capture(camera, face_recognizer, db_manager, alert_generator):
    frame = camera.get_frame()

    face_locations, face_names, face_types = face_recognizer.recognize_faces(frame)

    for (top, right, bottom, left), name, face_type in zip(face_locations, face_names, face_types):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            if name != "Unknown":
                
                #for NFR test setup
                #detected_times.append(datetime.datetime.now())

                ret, buffer = cv2.imencode('.jpg', frame)
                alert = Alert(face_type, buffer.tobytes(), str(datetime.datetime.now()), name )
                
                

                alert_generator.generate_alert(alert, db_manager)
