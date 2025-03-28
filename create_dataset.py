import os
import pickle
import mediapipe as mp
import cv2

# Disable TensorFlow Optimizations for Performance
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Filter only directories inside DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip files like .gitignore

    print(f"Processing images for class {dir_}...")

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img_file_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_file_path)

        if img is None:
            print(f"Warning: Could not read image {img_file_path}. Skipping...")
            continue  

        # ✅ Resize Image Before Processing (Fixes Freezing Issue)
        img_resized = cv2.resize(img, (256, 256))  # Resize to 256x256
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(img_rgb)
        except Exception as e:
            print(f"Error processing {img_file_path}: {e}")
            continue  

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
                    data_aux.append(x - min(x_))  # Normalize X
                    data_aux.append(y - min(y_))  # Normalize Y

            data.append(data_aux)
            labels.append(int(dir_))  # Convert label to int for ML training

# Save processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset successfully saved with {len(data)} samples.")
