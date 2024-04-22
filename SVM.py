import pandas as pd
import numpy as np
import glob
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

def process_files(files):
    aggregated_features = pd.DataFrame()
    for filename in files:
        print(f"Processing file: {filename}")
        df = pd.read_csv(filename)
        # Extract coordinates from strings
        df[['left_pupil_x', 'left_pupil_y']] = df['left_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        df[['right_pupil_x', 'right_pupil_y']] = df['right_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)

        # Aggregate features for each file
        df_aggregated = df.agg({
            'left_pupil_x': ['mean', 'std', 'max', 'min'],
            'left_pupil_y': ['mean', 'std', 'max', 'min'],
            'right_pupil_x': ['mean', 'std', 'max', 'min'],
            'right_pupil_y': ['mean', 'std', 'max', 'min']
        }).transpose()

        # Flatten the DataFrame and create multi-index columns
        df_aggregated = df_aggregated.unstack().to_frame().transpose()
        df_aggregated.columns = ['_'.join(map(str, col)) for col in df_aggregated.columns.values]

        # Add label information based on filename
        df_aggregated['label'] = int(os.path.basename(filename).split('_')[0])
        aggregated_features = pd.concat([aggregated_features, df_aggregated], ignore_index=True)

    return aggregated_features

def train_and_evaluate(data):
    X = data.drop('label', axis=1)
    y = data['label']
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = svm.SVC(kernel='rbf', gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)

def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Sensitivity (Recall): {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Confusion Matrix:\n{cm}")

# Path definition
real_data_path = 'GazeTracking/gaze_data/real_data'
synthetic_data_path = 'GazeTracking/gaze_data/synthetic_data'

# File collection
real_files = glob.glob(os.path.join(real_data_path, "*.csv"))
synthetic_files = glob.glob(os.path.join(synthetic_data_path, "*.csv"))

# Data processing
if real_files:
    print("Processing real data...")
    real_data = process_files(real_files)
    print("Training and evaluating SVM for real data...")
    train_and_evaluate(real_data)

if synthetic_files:
    print("Processing synthetic data...")
    synthetic_data = process_files(synthetic_files)
    print("Training and evaluating SVM for synthetic data...")
    train_and_evaluate(synthetic_data)

# Optionally combine data if both types are present
if real_files and synthetic_files:
    print("Processing combined data...")
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
    print("Training and evaluating SVM for combined data...")
    train_and_evaluate(combined_data)


print("Script completed.")

