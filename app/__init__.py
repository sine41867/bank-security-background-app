import cv2
import logging
import time
import datetime
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os

from app.models.alert_generator import AlertGenerator
from app.models.camera_handler import CameraHandler
from app.models.database_manager import DatabaseManager
from app.models.face_recognizer import FaceRecognizer
from app.models.alert import Alert

logging.basicConfig(
    filename='app.log',          
    level=logging.INFO,      
    format='%(asctime)s - %(levelname)s - %(message)s'
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

behavior_model = tf.keras.models.load_model(r'app\ai_models\model_abnormal_behaviour.h5')
fear_model = tf.keras.models.load_model(r'app\ai_models\model_fear.h5')

#net = cv2.dnn.readNetFromCaffe("app/utili/deploy.prototxt", "app/utili/mobilenet_iter_73000.caffemodel")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

db_manager = DatabaseManager()
alert_generator = AlertGenerator()

def app():
    
    known_faces, known_names, face_types = db_manager.load_known_faces()
    
    if not known_faces:
        
        logging.info('No known faces loaded from the database.')

    face_recognizer = FaceRecognizer(known_faces, known_names, face_types)

    cap = cv2.VideoCapture(0)
    
    interval_seconds = 2

    last_frame_time = time.time()
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()

            if current_time - last_frame_time >= interval_seconds:
                last_frame_time = current_time
                
                real_time_abnormal_detector_one_person(frame)
                real_time_face_detector(frame, face_recognizer)
                real_time_fear_detector(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                

        except Exception as e:
            logging.error(str(e))
            
    cap.release()


def real_time_abnormal_detector(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    people_count = 0
    abnormal_count = 0
    for i in range(detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            people_count +=1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x_max, y_max) = box.astype("int")

            person_roi = frame[y:y_max, x:x_max]

            pose_features = extract_pose_features(person_roi)

            if pose_features.any(): 
                pose_features_reshaped = pose_features.reshape(1, -1)  
                prediction = behavior_model.predict(pose_features_reshaped)

                is_normal = prediction[0][0] > 0.5 

                if is_normal:
                    label = "Normal"
                else:
                    label = "Abnormal"
                    abnormal_count += 1
                
                color = (0, 255, 0) if is_normal else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if people_count == 0 or abnormal_count == 0:
        return
    
    abnormal_percentage = abnormal_count*100/people_count

    print(f"Abnormal Percentage : {abnormal_percentage}")
    logging.info(f"Abnormal Percentage : {abnormal_percentage}")

    if abnormal_percentage >= 50:
            ret, buffer = cv2.imencode('.jpg', frame)
            alert = Alert("Abnormal", buffer.tobytes(), str(datetime.datetime.now()), (str(abnormal_percentage)+' %') )
            alert_generator.generate_alert(alert, db_manager)

def real_time_abnormal_detector_one_person(frame):

    feature_vector = extract_pose_features(frame)

    prediction = behavior_model.predict(np.expand_dims(feature_vector, axis=0))
    is_normal = prediction[0][0] > 0.5 

    if is_normal:
        return

    logging.info(f"Abnormal Behavior Detected")

    ret, buffer = cv2.imencode('.jpg', frame)
    alert = Alert("Abnormal", buffer.tobytes(), str(datetime.datetime.now()), "Abnormal behavior detected" )
    alert_generator.generate_alert(alert, db_manager)

def real_time_face_detector(frame, face_recognizer):
    
    face_locations, face_names, face_types = face_recognizer.recognize_faces(frame)

    for (top, right, bottom, left), name, face_type in zip(face_locations, face_names, face_types):
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            if name != "Unknown":

                ret, buffer = cv2.imencode('.jpg', frame)
                alert = Alert(face_type, buffer.tobytes(), str(datetime.datetime.now()), name )

                #testing
                print(f'{alert.alert_type}, {alert.description}, {alert.time}')

                alert_generator.generate_alert(alert, db_manager)
     
def real_time_fear_detector(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = 0
    fear_count = 0
    
    for (x, y, w, h) in faces:
        face_count += 1
        #print(f'face count : {face_count}')

        face_img = frame[y:y+h, x:x+w]
    
        face_img = preprocess_input_fear(face_img)
        face_img = np.expand_dims(face_img, axis=0)  
        prediction = fear_model.predict(face_img, verbose=0)

        if prediction[0][0] > 0.5:
            fear_count += 1
            #print(f'fear count : {fear_count}')
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "Fear", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

    fear_percentage = 0

    #if face_count > 5:
    if face_count:
        fear_percentage = fear_count*100/face_count
        #print(f'Final fear percentage : {fear_percentage}\n Final fear count : {fear_count}\n Final face count : {face_count}')

    if fear_percentage > 80:
        logging.info(f"{fear_percentage}% fear_percentage detected")
        ret, buffer = cv2.imencode('.jpg', frame)
        alert = Alert("Fear", buffer.tobytes(), str(datetime.datetime.now()), f"Fear in {fear_percentage}% faces" )
        alert_generator.generate_alert(alert, db_manager)

def extract_pose_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks.landmark])
        return landmarks.flatten() 
    else:
        return np.zeros(33 * 3) 
 
def preprocess_input_fear(frame):
    frame = tf.image.resize(frame, (224, 224)) 
    frame = tf.keras.applications.mobilenet_v3.preprocess_input(frame) 
    return frame


#to remove
def real_time_capture(camera, face_recognizer, db_manager, alert_generator):
    frame = camera.get_frame()

    face_locations, face_names, face_types = face_recognizer.recognize_faces(frame)

    for (top, right, bottom, left), name, face_type in zip(face_locations, face_names, face_types):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            if name != "Unknown":
               
                ret, buffer = cv2.imencode('.jpg', frame)
                alert = Alert(face_type, buffer.tobytes(), str(datetime.datetime.now()), name )
                
                alert_generator.generate_alert(alert, db_manager)

def app_v2():
    
    db_manager = DatabaseManager()
    
    known_faces, known_names, face_types = db_manager.load_known_faces()
    
    if not known_faces:
        logging.info('No known faces loaded from the database.')
        return

    face_recognizer = FaceRecognizer(known_faces, known_names, face_types)
    
    camera = CameraHandler()

    alert_generator = AlertGenerator()

    while True:
        try:
            real_time_capture(camera, face_recognizer, db_manager, alert_generator)
        except Exception as e:
             logging.error(str(e))
