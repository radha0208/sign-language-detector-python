import os
import pickle
import cv2
import mediapipe as mp

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Iterate through each gesture folder
for gesture_dir in os.listdir(DATA_DIR):
    gesture_path = os.path.join(DATA_DIR, gesture_dir)
    if not os.path.isdir(gesture_path):
        continue

    print(f"Processing gesture: {gesture_dir}")

    for img_file in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]

                # Normalize landmarks relative to the top-left of the hand bounding box
                data_aux = []
                min_x, min_y = min(x_list), min(y_list)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(gesture_dir)  # Keep gesture folder name as label

# Save processed dataset
output_file = 'data_landmarks.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created with {len(data)} samples for {len(set(labels))} classes!")
