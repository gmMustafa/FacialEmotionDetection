from flask import Flask, render_template, Response,request
from flask.json import jsonify
from keras.models import model_from_json
import cv2
import numpy as np
from threading import Thread
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
from flask_cors import CORS
import time
import logging
import dlib
import itertools
from keras.models import load_model
import mean_std_ofdata as ms


import os
import datetime


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode=None)
app.logger.setLevel(logging.DEBUG)

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# find mean and std of the dataset to normalize the input data
d_mean, d_std = ms.mean_std_of_data(r"./arc/facial_landmarks_with_distances_temp.csv")

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./arc/shape_predictor_68_face_landmarks.dat")

# Load the saved models
classifier = load_model(r"./arc/nn_model.h5")
chehra_pred=[]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')  # Render the About Us page


@app.route('/api_docs')
def api_docs():
    return render_template('api_docs.html')  # Render the API Docs page

# chehra model
def process_chehra_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global chehra_pred
    # Detect faces in the frame
    faces = detector(gray)
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        landmark_points = [num for num in range(17, 68) if num not in (60, 64)]
        # Extract X and Y coordinates for all 68 landmarks
        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_points]

        # Draw points on the face
        for (x, y) in landmarks_list:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Extract the region of interest (ROI) for the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        # Calculate distances between each pair of points
        distances = []
        for pair in itertools.combinations(landmarks_list, 2):
            x1, y1 = pair[0]
            x2, y2 = pair[1]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances.append(distance)
            
        # Convert distances to a numpy array and perform PCA
        distances_array = np.array(distances).reshape(1, -1)
        

        normalized_distances_array = (distances_array-d_mean)/d_std
        # Make a prediction using the SVM classifier
        prediction = classifier.predict(normalized_distances_array)
        chehra_pred=prediction
        # prediction for NN
        labels = ["anger","fear","happiness","neutrality","sadness","surprise"]
        prediction = [labels[prediction[0].argmax()]]
        # Display the prediction on the frame
        cv2.putText(frame, f"{prediction[0]}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    _, encoded_image = cv2.imencode('.jpg', frame)
    return encoded_image.tobytes()


@app.route('/process_chehra_frame', methods=['POST'])
def process_chehra_frame_route():
    frame_data = request.files['frame'].read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
            raise ValueError("Invalid frame data")
    processed_frame = process_chehra_frame(frame)
    return Response(response=processed_frame, content_type='image/jpeg')

@app.route('/get_chehra_predictions', methods=['GET'])
def get_chehra_predictions():
    try:
        global chehra_pred
         # Convert the NumPy array to a Python list
        # Convert the NumPy array to a Python list
        predictions_list = chehra_pred.tolist()
        response_data = {
            'predictions': predictions_list
        }
        return jsonify(response_data),200
    
    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 500



# CNN Model
json_file = open(r"./model/80perc/emotiondetector.json")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(r"./model/80perc/emotiondetector.h5")

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

emotion_info = {'anger': 0, 'contempt': 0, 'disgust': 0, 'fear': 0, 'happiness': 0, 'neutrality': 0, 'sadness': 0,
                'surprise': 0}
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
cnn_pred=[]




def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(-1, 48, 48, 1)
    return feature / 255.0

def process_CNN_frame(frame):
    global emotion_info,cnn_pred
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = extract_features(roi_gray)
        pred = model.predict(img)
        cnn_pred=pred
        cv2.rectangle(frame, (x, y), (x+w, y+h+10), (0, 255, 0), 2)
        label=np.argmax(pred,axis=1)[0]
        final_prediction = emotion_labels[label]
        cv2.putText(frame, final_prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    _, encoded_image = cv2.imencode('.jpg', frame)
    return encoded_image.tobytes()

@app.route('/process_CNN_frame', methods=['POST'])
def process_CNN_frame_route():
    frame_data = request.files['frame'].read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
            raise ValueError("Invalid frame data")
    processed_frame = process_CNN_frame(frame)
    return Response(response=processed_frame, content_type='image/jpeg')

@app.route('/get_cnn_predictions', methods=['GET'])
def get_cnn_predictions():
    try:
        global cnn_pred
         # Convert the NumPy array to a Python list
        # Convert the NumPy array to a Python list
        predictions_list = cnn_pred.tolist()
        response_data = {
            'predictions': predictions_list
        }
        return jsonify(response_data),200
    
    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 500


####Yollo Model######
# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
def yollo_model_config():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights(r"./model/model.h5")
    return model
yollo_pred=[]
def process_yollo_frame(frame):
    model=yollo_model_config()
    global yollo_pred
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    facecasc = cv2.CascadeClassifier(haar_file)
    # facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    np.set_printoptions(suppress=True)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Ensure your input data is scaled/normalized in the same way as during training
        # If your model was trained with images scaled to [0,1], ensure the same here:
        cropped_img = cropped_img / 255.0

        predictions = model.predict(cropped_img)
        max_index = int(np.argmax(predictions))
        confidence_scores = predictions[0]
        
        print(f"{emotion_dict}: {confidence_scores.tolist()}")
        yollo_pred=confidence_scores.tolist()

        # Optionally, you can display the emotion with the highest score on the image window
        
        cv2.putText(frame, emotion_dict[max_index], (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    _, encoded_image = cv2.imencode('.jpg', frame)
    return encoded_image.tobytes()


@app.route('/process_yollo_frame', methods=['POST'])
def process_Yollo_frame_route():
    frame_data = request.files['frame'].read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
            raise ValueError("Invalid frame data")
    processed_frame = process_yollo_frame(frame)
    return Response(response=processed_frame, content_type='image/jpeg')


@app.route('/get_yollo_predictions', methods=['GET'])
def get_yollo_predictions():
    try:
        global yollo_pred
        response_data = {
            'predictions': yollo_pred
        }
        return jsonify(response_data),200
    
    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 500




#####################################################
def create_folder_and_file(session_id, model_name, content):
    # Create a timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    session_id= "id_"+session_id
    base_dir = "./sessions-data/"
    folder_name = f"{session_id}_{timestamp}"
    file_name = f"{model_name}_{timestamp}.txt"
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Full path for the new folder
    full_folder_path = os.path.join(base_dir, folder_name)

    # Create the folder
    os.makedirs(full_folder_path, exist_ok=True)

    # Create and write to the file
    file_path = os.path.join(full_folder_path, file_name)
    with open(file_path, 'w') as file:
        file.write(content)  # Replace with your actual content

    return f"File {file_name} created in folder {folder_name}"

# Flask route
@app.route('/create_file/<session_id>/<model_name>', methods=['POST'])
def create_file_route(session_id, model_name):
    content = request.json.get('content', '')  # Get content from JSON
    result = create_folder_and_file(session_id, model_name, content)
    return result
#####################################################    


if __name__ == '__main__':
    socketio.run(app, debug=True)
