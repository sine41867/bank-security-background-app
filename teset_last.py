import cv2
import tensorflow as tf
import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

fear_model = tf.keras.models.load_model(r'app\ai_models\model_fear.h5')

face_cascade_path = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

def preprocess_input_fear(frame):
    frame = tf.image.resize(frame, (224, 224)) 
    frame = tf.keras.applications.mobilenet_v3.preprocess_input(frame) 
    return frame

def real_time_fear_detector(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = 0
    fear_count = 0
    
    for (x, y, w, h) in faces:
        face_count += 1
        print(f'face count : {face_count}')
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Predict emotion
        face_img = preprocess_input_fear(face_img)
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
        prediction = fear_model.predict(face_img, verbose=0)

        if prediction[0][0] > 0.5:
            fear_count += 1
            print(f'fear count : {fear_count}')
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "Fear", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

    fear_percentage = 0

    if face_count:
        fear_percentage = fear_count*100/face_count
        print(f'Final fear percentage : {fear_percentage}\n Final fear count : {fear_count}\n Final face count : {face_count}')

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
            
            real_time_fear_detector(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            

    except Exception as e:
        print(str(e))
        
cap.release()
