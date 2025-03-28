import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import base64

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Dictionary mapping for A-Z & 0-9
labels_dict = {i: chr(65 + i) if i < 26 else str(i - 26) for i in range(36)}

# Function to process frame and predict
def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:  # 21 landmarks * 2 (x, y)
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "?")
            return predicted_character
    return None

# Route for the homepage
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Route to process video frames
@app.route('/predict', methods=['POST'])
def predict():
    # Get the frame from the request (sent as base64)
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    # Decode base64 frame
    frame_data = base64.b64decode(data['frame'].split(',')[1])
    np_frame = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Process the frame and get prediction
    prediction = process_frame(frame)

    if prediction:
        return jsonify({'prediction': prediction})
    return jsonify({'prediction': ''})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
