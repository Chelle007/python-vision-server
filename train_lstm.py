import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# These are the actual outputs your AI will guess in the game
FINAL_CLASSES = ["Idle", "Turn_Key", "Pull_Lever"] 

# This maps your human-organized folders to the AI classes
FOLDER_MAPPING = {
    "Idle": "Idle",                # Normal idle/waving
    "FP_Turn_Key": "Idle",           # Bad key attempts -> map to Idle
    "FP_Pull_Lever": "Idle",         # Bad lever attempts -> map to Idle
    "Turn_Key": "Turn_Key",        # Correct key turns
    "Pull_Lever": "Pull_Lever"     # Correct lever pulls
}

DATASET_PATH = "Dataset"
NUM_FRAMES = 30
NUM_FEATURES = 63 # 21 landmarks * (x, y, z)

# ==========================================
# 2. LOAD & COMBINE DATA
# ==========================================
print("--- STARTING DATA LOADING ---")
sequences, labels = [], []

for folder_name, target_class in FOLDER_MAPPING.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Skipping: {folder_path} (Folder not found)")
        continue
        
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"Loading {len(filenames)} files from '{folder_name}' into class '{target_class}'...")
    
    label_id = FINAL_CLASSES.index(target_class)
    
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        res = np.load(file_path)
        # Basic validation to ensure frame count matches
        if res.shape == (NUM_FRAMES, NUM_FEATURES):
            sequences.append(res)
            labels.append(label_id)

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(FINAL_CLASSES)).astype(int)

print(f"\nTOTAL SAMPLES LOADED: {X.shape[0]}")
print(f"INPUT DATA SHAPE: {X.shape} (Samples, Frames, Features)")

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD THE LSTM NEURAL NETWORK
# ==========================================
print("\n--- BUILDING MODEL ---")
model = Sequential()

# Layer 1: LSTM with 64 units
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(NUM_FRAMES, NUM_FEATURES)))
# Layer 2: LSTM with 32 units (returns final state)
model.add(LSTM(32, return_sequences=False, activation='relu'))
# Dropout: Randomly ignores 20% of neurons to prevent "memorizing" data
model.add(Dropout(0.2))
# Dense Layer: Final classification
model.add(Dense(len(FINAL_CLASSES), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. TRAINING
# ==========================================
print("\n--- TRAINING STARTED ---")
# Epochs 50-70 is usually the sweet spot for this size of dataset
history = model.fit(
    X_train, y_train, 
    epochs=60, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# ==========================================
# 5. SAVE RESULT
# ==========================================
MODEL_NAME = "escape_gestures.keras"
model.save(MODEL_NAME)
print(f"\nSUCCESS! Model saved as {MODEL_NAME}")