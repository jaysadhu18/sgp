import os
import cv2

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 36
dataset_size = 100

cap = cv2.VideoCapture(0)  # Try changing to 1 if using an external webcam
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')
    
    # Wait for user confirmation
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Couldn't read frame. Check camera connection.")
            continue
        
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Skipping frame due to capture failure.")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1
        if cv2.waitKey(50) & 0xFF == ord('q'):  # Allow stopping mid-capture
            break

cap.release()
cv2.destroyAllWindows()
