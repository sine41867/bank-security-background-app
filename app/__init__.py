import cv2
import logging
import time
import datetime
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import torch
from PIL import Image
import io



from app.models.alert_generator import AlertGenerator
from app.models.database_manager import DatabaseManager
from app.models.alert import Alert

logging.basicConfig(
    filename='app.log',          
    level=logging.INFO,      
    format='%(asctime)s - %(levelname)s - %(message)s'
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

behavior_model = tf.keras.models.load_model(r'app\ai_models\model_abnormal_behaviour.h5')
fear_model = tf.keras.models.load_model(r'app\ai_models\model_fear.h5')

net = cv2.dnn.readNetFromCaffe("app/utili/deploy.prototxt", "app/utili/mobilenet_iter_73000.caffemodel")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mtcnn = MTCNN(keep_all=True)
inception_resnet = InceptionResnetV1(pretrained='vggface2').eval()

face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

db_manager = DatabaseManager()
alert_generator = AlertGenerator()

def app():
    
    global known_persons
    known_persons = load_face_emebeddings()

    if not known_persons:
        
        logging.info('No known persons loaded from the database.')

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
                
                real_time_fear_detector(frame)

                real_time_face_recognizer_test(frame)

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

    if face_count > 3:
    #if face_count:
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


def real_time_face_recognizer_test(frame):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    embeddings, faces = get_embedding(image_rgb)

    if faces is None or embeddings is None:
        return
    
    for i, face in enumerate(faces):
        cv2.rectangle(frame, 
                            (int(face[0]), int(face[1])), 
                            (int(face[2]), int(face[3])), 
                            (0, 255, 0), 2)  # Draw bounding box around detected face

        if len(known_persons) == 0:
            return

        for embedding, name, face_type in known_persons:
            for emb, face in zip(embeddings, faces):  # Compare all detected faces with known faces
                match, match_percentage = compare_faces(emb, embedding)

                if match and match_percentage > 70:
                    cv2.putText(frame, f"{face_type} : {name}", 
                                (int(face[0]), int(face[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    recognized = True

                    logging.info(f"Recognized: {face_type} {name} with match: {match_percentage:.2f}%")

                    ret, buffer = cv2.imencode('.jpg', frame)
                    alert = Alert(face_type, buffer.tobytes(), str(datetime.datetime.now()), name )

                    alert_generator.generate_alert(alert, db_manager)
            
def load_face_emebeddings():

    loaded_faces = db_manager.load_images()

    known_faces = []


    for item in loaded_faces:
       
        image = Image.open(io.BytesIO(item[0]))
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_array = np.array(image)
      
        embeddings, _ = get_embedding(image_array)
        

        if embeddings is not None:
            for embedding in embeddings:
                
                known_faces.append((embedding,item[1],item[2]))
            
    return known_faces
        
        

def get_embedding(image):
    
    faces, _ = mtcnn.detect(image)
    if faces is not None:
        embeddings = inception_resnet(mtcnn(image))  # Extract embeddings
        return embeddings, faces  # Return both embeddings and face bounding boxes
    return None, None

def compare_faces(embedding1, embedding2, threshold=0.6):
    """Compare two face embeddings using cosine distance and return the match percentage."""
    
    # If embeddings are tensors, detach and convert to NumPy arrays
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.detach().cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.detach().cpu().numpy()

    # Compute cosine distance between the two embeddings
    distance = cosine(embedding1, embedding2)
    match_percentage = (1 - distance) * 100  # Convert distance to match percentage
    #print(f"Distance between embeddings: {distance}, Match Percentage: {match_percentage:.2f}%")
    
    if distance < threshold:
        return True, match_percentage  # Return match status and percentage
    return False, match_percentage  # Return match status and percentage
