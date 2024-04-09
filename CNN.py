import os
import glob
import pandas as pd
import numpy as np
import re

# Suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input, Dropout, BatchNormalization, GaussianNoise
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam


# Define paths to the data
path_real = r'GazeTracking/gaze_data/real_data/*.csv'
path_synthetic = r'GazeTracking/gaze_data/synthetic_data/*.csv'

def load_data_and_labels(path):
    files = glob.glob(path)
    sequences = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        label = int(re.search(r'^(\d)_', file.split('\\')[-1]).group(1))
        
        # Initialize new columns to ensure they exist
        df['left_pupil_x'] = np.nan
        df['left_pupil_y'] = np.nan
        df['right_pupil_x'] = np.nan
        df['right_pupil_y'] = np.nan

        # Extract coordinates
        for column in ['left_pupil', 'right_pupil']:
            coords = df[column].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
            df[f'{column}_x'] = coords[0]
            df[f'{column}_y'] = coords[1]

        # Remove rows where extraction failed (any NaN in crucial columns)
        df = df.dropna(subset=['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y'])

        if df.empty:
            continue  # Skip this file if no valid rows remain

        encoder = LabelEncoder()
        df['gaze_direction_encoded'] = encoder.fit_transform(df['gaze_direction'])
        features = df[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'gaze_direction_encoded']].values

        sequences.append(features)
        labels.append(label)

    return sequences, labels

sequences_real, labels_real = load_data_and_labels(path_real)
sequences_synthetic, labels_synthetic = load_data_and_labels(path_synthetic)

sequences = sequences_real + sequences_synthetic
labels = labels_real + labels_synthetic

from keras.preprocessing.sequence import pad_sequences
sequences_padded = pad_sequences(sequences, padding='post', dtype='float32')
labels_encoded = to_categorical(labels)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train, test in kfold.split(sequences_padded, labels_encoded):
    scaler = StandardScaler()
    X_train_reshaped = sequences_padded[train].reshape(-1, sequences_padded[train].shape[-1])
    X_test_reshaped = sequences_padded[test].reshape(-1, sequences_padded[test].shape[-1])
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(sequences_padded[train].shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(sequences_padded[test].shape)
    
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        GaussianNoise(0.01),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(50, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train_scaled, labels_encoded[train], epochs=100, validation_data=(X_test_scaled, labels_encoded[test]), callbacks=[early_stopping], batch_size=32)

    predictions = model.predict(X_test_scaled)
    predictions_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels_encoded[test], axis=1)

    accuracy_scores.append(np.mean(predictions_labels == true_labels))
    precision_scores.append(precision_score(true_labels, predictions_labels, average='macro', zero_division=0))
    recall_scores.append(recall_score(true_labels, predictions_labels, average='macro', zero_division=0))
    f1_scores.append(f1_score(true_labels, predictions_labels, average='macro', zero_division=0))

print(f"Average Accuracy: {round(np.mean(accuracy_scores) * 100, 2)}%")
print(f"Average Precision: {round(np.mean(precision_scores) * 100, 2)}%")
print(f"Average Sensitivity (Recall): {round(np.mean(recall_scores) * 100, 2)}%")
print(f"Average F1 Score: {round(np.mean(f1_scores) * 100, 2)}%")
