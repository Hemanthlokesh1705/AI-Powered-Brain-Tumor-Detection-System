import os
import numpy as np
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('TkAgg') 


# Set dataset paths
DATA_DIR = r"AI powered Brain Tumor Detection System\photos"  # Update with your dataset path
TUMOR_DIR = os.path.join(DATA_DIR, "yes")  # Tumor images
NON_TUMOR_DIR = os.path.join(DATA_DIR, "no")  # No tumor images

# Load and preprocess images
X, Y = [], []

for label, folder in enumerate([NON_TUMOR_DIR, TUMOR_DIR]):  # Changed order: 0=No Tumor, 1=Tumor
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:  # Check if image was loaded successfully
            img = cv2.resize(img, (128, 128))  # Resize for CNN
            X.append(img)
            Y.append(label)
        else:
            print(f"Warning: Could not load image {img_path}")

# Convert to NumPy arrays and normalize
X = np.array(X) / 255.0  # Normalize
X = X.reshape(-1, 128, 128, 1)  # Add channel dimension
Y = np.array(Y)

# Split into train & test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, Y_train, 
                   validation_data=(X_test, Y_test), 
                   epochs=10, 
                   batch_size=32,
                   callbacks=[callback])

# Save the trained model
model.save("brain_tumor_model.h5")

# Save the model history (for plotting)
with open("train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.savefig('training_history.png')
plt.show()