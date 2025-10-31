import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100  # default number of images per gesture

# Predefined classes
alpha_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def collect_gesture_images(gesture_name, dataset_size=100):
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    
    print(f"\nCollecting data for: {gesture_name}")
    print("Press 'Q' when ready to start...")

    # Wait for user to press Q
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Frame", frame)
        cv2.imwrite(os.path.join(gesture_dir, f"{counter}.jpg"), frame)
        counter += 1

        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):  # press 'S' to stop collecting this gesture
            print(f"Stopped collecting for {gesture_name}")
            break
        if counter >= dataset_size:
            print(f"Reached default dataset size for {gesture_name}")
            break

# --- WORD COLLECTION ---
collect_words = input("Do you want to collect word gestures? (y/n): ").lower()
if collect_words == 'y':
    while True:
        word_name = input("Enter word class name (or 'stop' to end words): ")
        if word_name.lower() == 'stop':
            break
        collect_gesture_images(f"word_{word_name}", dataset_size)

# --- ALPHABET COLLECTION ---
collect_alpha = input("Do you want to collect alphabet gestures? (y/n): ").lower()
if collect_alpha == 'y':
    while True:
        alpha_letter = input("Enter alphabet letter (or 'stop' to end alphabets): ").upper()
        if alpha_letter.lower() == 'stop':
            break
        if alpha_letter not in alpha_classes:
            print("Invalid letter, try again.")
            continue
        collect_gesture_images(f"alpha_{alpha_letter}", dataset_size)

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
