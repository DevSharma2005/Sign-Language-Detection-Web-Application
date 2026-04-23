import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silences Info and Warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Specifically turns off the oneDNN message

import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# The signs you want to train
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
          "V", "W", "X", "Y", "Z", "HELLO", "HOW ARE YOU", "QUIT"]python
samples_per_label = 100

cap = cv2.VideoCapture(0)

with open('data/hand_data.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    for idx, label_name in enumerate(labels):
        print(f"Get ready to sign: {label_name}")
        time.sleep(3) # Give you time to position your hand
        
        count = 0
        while count < samples_per_label:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # NORMALIZATION LOGIC (The Secret Sauce)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z])
                
                writer.writerow(landmarks + [idx])
                count += 1
                
                # Visual Feedback
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Capturing {label_name}: {count}/100", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Data Generator', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(f"Done with {label_name}!")

cap.release()
cv2.destroyAllWindows()