import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

behavior_model = keras.models.load_model(r'app\ai_models\model_abnormal_behaviour.keras')

net = cv2.dnn.readNetFromCaffe("app/utili/deploy.prototxt", "app/utili/mobilenet_iter_73000.caffemodel")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_pose_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks.landmark])
        return landmarks.flatten() 
    else:
        return np.zeros(33 * 3) 

def tester_v1():
   
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5: 
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x_max, y_max) = box.astype("int")

                    person_roi = frame[y:y_max, x:x_max]

                    pose_features = extract_pose_features(person_roi)

                    if pose_features.any(): 
                        pose_features_reshaped = pose_features.reshape(1, -1)  
                        prediction = behavior_model.predict(pose_features_reshaped)

                        is_normal = prediction[0][0] > 0.5  
                        label = "Normal" if is_normal else "Abormal"
                        
                        color = (0, 255, 0) if is_normal else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Abnormal Behavior Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(str(e))
            
    cap.release()
    cv2.destroyAllWindows()
