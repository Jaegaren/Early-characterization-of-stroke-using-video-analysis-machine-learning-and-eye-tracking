import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load data from real and synthetic paths
path_to_real_data = r'GazeTracking/gaze_data/real_data/'
path_to_synthetic_data = r'GazeTracking/gaze_data/synthetic_data/'

all_files_real = glob.glob(path_to_real_data + "/*.csv")
all_files_synthetic = glob.glob(path_to_synthetic_data + "/*.csv")


def process_files(files):
    li = []
    for filename in files:
        label = int(os.path.basename(filename).split('_')[0])
        df = pd.read_csv(filename, index_col=None, header=0)
        df['label'] = label

        # Preprocess pupil coordinates
        df['left_pupil_x'] = df['left_pupil'].str.extract(r'\((.*),')[0].astype(float)
        df['left_pupil_y'] = df['left_pupil'].str.extract(r', (.*)\)')[0].astype(float)
        df['right_pupil_x'] = df['right_pupil'].str.extract(r'\((.*),')[0].astype(float)
        df['right_pupil_y'] = df['right_pupil'].str.extract(r', (.*)\)')[0].astype(float)

        li.append(df)
    if li:
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


# Process real and synthetic data separately
real_data = process_files(all_files_real)
synthetic_data = process_files(all_files_synthetic)

# Combine real and synthetic data
if real_data is not None and synthetic_data is not None:
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
else:
    combined_data = real_data if real_data is not None else synthetic_data

# Select features and labels
X = combined_data[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]
y = combined_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the DNN Model
def build_dnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Define input shape and number of classes
input_shape = (X_train.shape[1],)
num_classes = len(np.unique(y))
print(num_classes)

# Build the model
model = build_dnn_model(input_shape, num_classes)

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Make Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate Precision, Recall and F1 Score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {round(accuracy * 100, 2)}%")
print(f"Precision: {round(precision * 100, 2)}%")
print(f"Sensitivity (Recall): {round(recall * 100, 2)}%")
print(f"F1 Score: {round(f1 * 100, 2)}%")
