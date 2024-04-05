import os
import glob
import pandas as pd
import numpy as np
import re
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping

# Suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define path to the data
path_real = r'GazeTracking/gaze_data/real_data/*.csv'
path_synthetic = r'GazeTracking/gaze_data/synthetic_data/*.csv'

def load_data_and_labels(path):
    files = glob.glob(path)
    sequences = []
    labels = []
    
    for file in files:
        df = pd.read_csv(file)
        label = int(re.search(r'^(\d)_', file.split('\\')[-1]).group(1))
        
        # Extract and convert coordinates
        for column in ['left_pupil', 'right_pupil']:
            df[[f'{column}_x', f'{column}_y']] = df[column].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        
        encoder = LabelEncoder()
        df['gaze_direction_encoded'] = encoder.fit_transform(df['gaze_direction'])
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(df[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'gaze_direction_encoded']])
        
        sequences.append(features)
        labels.append(label)
        
    return sequences, labels

sequences_real, labels_real = load_data_and_labels(path_real)
sequences_synthetic, labels_synthetic = load_data_and_labels(path_synthetic)

sequences = sequences_real + sequences_synthetic
labels = labels_real + labels_synthetic

# Padding sequences and one-hot encoding labels
from keras.preprocessing.sequence import pad_sequences
sequences_padded = pad_sequences(sequences, padding='post', dtype='float32')
labels_encoded = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_encoded, test_size=0.2, random_state=42)

# Building the model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(50, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

predictions = model.predict(X_test)
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions_labels == true_labels)
precision = precision_score(true_labels, predictions_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predictions_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predictions_labels, average='macro', zero_division=0)

print(f"Accuracy: {round(accuracy * 100, 2)}%")
print(f"Precision: {round(precision * 100, 2)}%")
print(f"Sensitivity (Recall): {round(recall * 100, 2)}%")
print(f"F1 Score: {round(f1 * 100, 2)}%")
