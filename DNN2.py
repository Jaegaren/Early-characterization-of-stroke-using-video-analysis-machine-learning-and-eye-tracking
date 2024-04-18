import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import Callback

# Loading and preprocessing the data by calculating difference in coordinates for each frame
def load_and_preprocess_data(folder_path):
    differences = []  # This will store the coordinate differences for each file
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Extract label from the first character of the file name
            label = int(file_name[0])

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, usecols=['left_pupil', 'right_pupil'])

            # Drop rows with missing values
            df = df.dropna()

            # Calculate coordinate differences
            file_differences = []
            previous_row = None
            for _, row in df.iterrows():
                current_row = {
                    'left_pupil': eval(row['left_pupil']),
                    'right_pupil': eval(row['right_pupil'])
                }

                if previous_row is not None:
                    # Calculate differences for left pupil
                    left_diff = (current_row['left_pupil'][0] - previous_row['left_pupil'][0],
                                 current_row['left_pupil'][1] - previous_row['left_pupil'][1])

                    # Calculate differences for right pupil
                    right_diff = (current_row['right_pupil'][0] - previous_row['right_pupil'][0],
                                  current_row['right_pupil'][1] - previous_row['right_pupil'][1])

                    file_differences.append(left_diff + right_diff)

                # Update the previous row
                previous_row = current_row

            differences.append(file_differences)
            labels.append(label)

    return differences, labels

# Load data from real and synthetic paths
path_to_real_data = r'GazeTracking/gaze_data/real_data/'
path_to_synthetic_data = r'GazeTracking/gaze_data/synthetic_data/'

all_files_real = glob.glob(path_to_real_data + "/*.csv")
all_files_synthetic = glob.glob(path_to_synthetic_data + "/*.csv")

# Process real and synthetic data separately
real_diff, real_labels = load_and_preprocess_data(path_to_real_data)
synthetic_diff, synthetic_labels = load_and_preprocess_data(path_to_synthetic_data)

# Combine real and synthetic differences and labels
all_diff = real_diff + synthetic_diff
all_labels = real_labels + synthetic_labels

# Padding sequences to the same length
max_len = max(len(seq) for seq in all_diff)
padded_diff = [seq + [(0, 0, 0, 0)] * (max_len - len(seq)) for seq in all_diff]

# Convert differences and labels to numpy arrays
X_diff = np.array(padded_diff)
y_diff = np.array(all_labels)

# Split the data into training and testing sets
X_diff_train, X_diff_test, y_diff_train, y_diff_test = train_test_split(X_diff, y_diff, test_size=0.2, random_state=42)

# Flatten the differences array
X_diff_train_flat = np.reshape(X_diff_train, (X_diff_train.shape[0], -1))
X_diff_test_flat = np.reshape(X_diff_test, (X_diff_test.shape[0], -1))

# Define input shape for the differences
input_shape_diff = (X_diff_train_flat.shape[1],)

# Define input shape and number of classes for the combined model
num_classes = len(np.unique(y_diff))

# Define a simple DNN model for the differences
model_diff = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape_diff),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the differences model
model_diff.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train the differences model
model_diff.fit(X_diff_train_flat, y_diff_train, epochs=100, batch_size=32, validation_data=(X_diff_test_flat, y_diff_test))

# Evaluate the differences model
test_loss_diff, test_accuracy_diff = model_diff.evaluate(X_diff_test_flat, y_diff_test)
print("Test Accuracy (Differences Model):", test_accuracy_diff)

# Make predictions using the differences model
y_pred_diff = np.argmax(model_diff.predict(X_diff_test_flat), axis=1)

# Calculate Precision, Recall, and F1 Score for the differences model
precision_diff = precision_score(y_diff_test, y_pred_diff, average='macro')
recall_diff = recall_score(y_diff_test, y_pred_diff, average='macro')
accuracy_diff = accuracy_score(y_diff_test, y_pred_diff)
f1_diff = f1_score(y_diff_test, y_pred_diff, average='macro')

print(f"Accuracy (Differences Model): {round(accuracy_diff * 100, 2)}%")
print(f"Precision (Differences Model): {round(precision_diff * 100, 2)}%")
print(f"Sensitivity (Recall) (Differences Model): {round(recall_diff * 100, 2)}%")
print(f"F1 Score (Differences Model): {round(f1_diff * 100, 2)}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_diff_test, y_pred_diff)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_diff), yticklabels=np.unique(y_diff))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


