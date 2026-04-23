import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import pickle # To save the scaler for app.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load the data
data = pd.read_csv('data/hand_data.csv', header=None)
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

# --- NEW: Data Scaling ---
# This ensures a small movement isn't ignored by the AI
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler so app.py can use the exact same math
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# -------------------------

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 3. Enhanced Architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.2),
    Dense(256, activation='relu'), # Increased capacity to distinguish A vs B
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 4. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Training started on {len(X)} samples...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the model
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/sign_model.h5')
print("Model and Scaler saved successfully!")