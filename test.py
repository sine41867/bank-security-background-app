import cv2
import tensorflow as tf
import numpy as np
import os
import json
from keras._tf_keras.keras.models import model_from_json
from keras._tf_keras.keras.layers import BatchNormalization
from keras._tf_keras.keras.utils import custom_object_scope
from keras._tf_keras.keras.models import load_model
#from models import model_from_json

'''
# Load the model configuration
with open('model_config.json', 'r') as f:
    model_config = json.load(f)

# Modify the name of BatchNormalization layers
for layer in model_config['config']['layers']:
    if layer['class_name'] == 'BatchNormalization':
        if 'name' in layer['config'] and '/' in layer['config']['name']:
            layer['config']['name'] = layer['config']['name'].replace('/', '_')

# Save the updated configuration back to a file (if needed)
with open('updated_model_config.json', 'w') as f:
    json.dump(model_config, f)

# Load the model with the updated configuration
model = model_from_json(json.dumps(model_config))

# If you have weights to load, you can do so here
# model.load_weights('path_to_weights.h5')

'''
# Suppress TensorFlow logging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged (default behavior), 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

def CustomBatchNormalization(*args, **kwargs):
    kwargs['name'] = kwargs['name'].replace('/', '_')  # Replace `/` with `_`
    return BatchNormalization(*args, **kwargs)


with custom_object_scope({'TFOpLambda': tf.keras.layers.Lambda, 'BatchNormalization': CustomBatchNormalization}):
    model = load_model(r'app\ai_models\model_fear.h5')


# Load the trained model
#model = tf.keras.models.load_model(r'app\ai_models\model_fear.h5')

# Define paths
face_cascade_path = 'haarcascade_frontalface_default.xml'

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

# Preprocessing function to match the model's input requirements
def preprocess_input_fn(img):
    img = tf.image.resize(img, (224, 224))  # Resize to 224x224
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)  # MobileNetV3 preprocessing
    return img

# Function to predict emotion
def predict_emotion(face_img):
    face_img = preprocess_input_fn(face_img)
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    prediction = model.predict(face_img, verbose=0)  # Suppress verbose output
    return 'Fear' if prediction[0][0] > 0.5 else 'No Fear'

# Start video capture from a video. Check whether the path is correct
cap = cv2.VideoCapture(0)

# Start video capture from webcam
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        # Predict emotion
        emotion = predict_emotion(face)
        
        # Draw rectangle around face and display prediction
        color = (0, 255, 0) if emotion == 'No Fear' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
