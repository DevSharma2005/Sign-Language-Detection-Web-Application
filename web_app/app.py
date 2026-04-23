import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import threading
import time
import signal
from flask import Flask, render_template, Response

app = Flask(__name__)

# --- SPEECH SETUP ---
last_spoken = ""
last_speak_time = 0

def speak(text):
    """Initializes a fresh engine instance per thread for stability."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# --- MODEL LOADING ---
model = tf.keras.models.load_model('models/sign_model.h5')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
          "V", "W", "X", "Y", "Z", "HELLO", "HOW ARE YOU","QUIT"]

def generate_frames():
    global last_spoken, last_speak_time
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        prediction_text = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Landmark Math
                base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z])
                
                # 2. Prediction
                input_data = scaler.transform([landmarks])
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > 0.90:
                    current_sign = LABELS[class_id]
                    prediction_text = f"{current_sign} ({int(confidence*100)}%)"

                    # 3. Shutdown Logic
                    if current_sign == "QUIT":
                        os.kill(os.getpid(), signal.SIGINT)

                    # 4. SMART SPEECH TRIGGER
                    current_time = time.time()
                    # Trigger if sign changed OR if the same sign is held for 3 seconds
                    if current_sign != last_spoken or (current_time - last_speak_time > 3.0):
                        last_spoken = current_sign
                        last_speak_time = current_time
                        threading.Thread(target=speak, args=(current_sign,), daemon=True).start()
                
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # --- THE FIX IS HERE ---
            # If no hand is detected, we "forget" the last sign.
            # This allows the system to speak "Hello" then "A" then "Hello" again.
            last_spoken = "" 

        cv2.putText(frame, prediction_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)