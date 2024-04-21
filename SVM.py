import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import glob
import os

def process_files(files):
    li = []
    for filename in files:
        label = int(os.path.basename(filename).split('_')[0])
        print(f"Processing file: {filename}, Label: {label}")
        df = pd.read_csv(filename, index_col=None, header=0)
        df['label'] = label
        df[['left_pupil_x', 'left_pupil_y']] = df['left_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        df[['right_pupil_x', 'right_pupil_y']] = df['right_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    # Compute the mean of the numeric columns
    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_cols] = frame[numeric_cols].fillna(frame[numeric_cols].mean())
    return frame

# Define the paths
real_data_path = os.path.join('GazeTracking', 'gaze_data', 'real_data')
synthetic_data_path = os.path.join('GazeTracking', 'gaze_data', 'synthetic_data')

# Get the list of all files in each directory
real_files = glob.glob(os.path.join(real_data_path, "*.csv"))
synthetic_files = glob.glob(os.path.join(synthetic_data_path, "*.csv"))

print(f"Found {len(real_files)} real files.")
print(f"Found {len(synthetic_files)} synthetic files.")

# Process real data
if real_files:
    print("Processing real data...")
    real_frame = process_files(real_files)
    print("Real data processing completed.\n")

# Process synthetic data
if synthetic_files:
    print("Processing synthetic data...")
    synthetic_frame = process_files(synthetic_files)
    print("Synthetic data processing completed.\n")

# Combine real and synthetic data if both are present
if real_files and synthetic_files:
    print("Combining real and synthetic data...")
    combined_frame = pd.concat([real_frame, synthetic_frame], axis=0, ignore_index=True)

# Train and evaluate SVM for real, synthetic, and combined datasets
def train_and_evaluate_svm(X, y):
    test_sizes = [0.2, 0.25, 0.3]
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        print(f"Results for test size {test_size * 100}%:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Sensitivity (Recall): {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print(f"Confusion Matrix:\n{cm}\n")

if real_files:
    print("Training and evaluating SVM for real data...")
    X_real = real_frame[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]
    y_real = real_frame['label']
    train_and_evaluate_svm(X_real, y_real)

if synthetic_files:
    print("Training and evaluating SVM for synthetic data...")
    X_synthetic = synthetic_frame[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]
    y_synthetic = synthetic_frame['label']
    train_and_evaluate_svm(X_synthetic, y_synthetic)

if real_files and synthetic_files:
    print("Training and evaluating SVM for combined data...")
    X_combined = combined_frame[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]
    y_combined = combined_frame['label']
    train_and_evaluate_svm(X_combined, y_combined)

print("Script completed.")

