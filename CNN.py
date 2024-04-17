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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences


# Define paths to the data
path_real = r'GazeTracking/gaze_data/real_data/*.csv'
path_synthetic = r'GazeTracking/gaze_data/synthetic_data/*.csv'

def load_data_and_labels(path):
    files = glob.glob(path)
    all_differences = []
    labels = []
        
    for file in files:
        df = pd.read_csv(file)
        label = int(re.search(r'^(\d)_', file.split('\\')[-1]).group(1))
        
        # Initialize columns for coordinates
        df['left_pupil_x'] = np.nan
        df['left_pupil_y'] = np.nan
        df['right_pupil_x'] = np.nan
        df['right_pupil_y'] = np.nan

        # Extract and convert coordinates
        for column in ['left_pupil', 'right_pupil']:
            coords = df[column].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
            df[f'{column}_x'] = coords[0]
            df[f'{column}_y'] = coords[1]

        df = df.dropna(subset=['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y'])

        if df.empty:
            continue

        # Calculate differences in coordinates frame by frame
        differences = []
        previous_row = df.iloc[0]
        for index, row in df.iterrows():
            if index == 0:
                continue
            left_diff_x = row['left_pupil_x'] - previous_row['left_pupil_x']
            left_diff_y = row['left_pupil_y'] - previous_row['left_pupil_y']
            right_diff_x = row['right_pupil_x'] - previous_row['right_pupil_x']
            right_diff_y = row['right_pupil_y'] - previous_row['right_pupil_y']
            differences.append([left_diff_x, left_diff_y, right_diff_x, right_diff_y])
            previous_row = row

        all_differences.append(differences)
        labels.append(label)

    return all_differences, labels

sequences_real, labels_real = load_data_and_labels(path_real)
sequences_synthetic, labels_synthetic = load_data_and_labels(path_synthetic)

sequences = sequences_real + sequences_synthetic
labels = labels_real + labels_synthetic

# Pad sequences to ensure they are all the same length
sequences_padded = pad_sequences(sequences, padding='post', dtype='float32')
labels_encoded = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_encoded, test_size=0.2, random_state=42)

# Define and compile the CNN model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
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

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], batch_size=32)

# Evaluate the model
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
