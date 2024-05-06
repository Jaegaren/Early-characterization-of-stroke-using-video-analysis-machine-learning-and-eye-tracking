import pandas as pd
import numpy as np
import glob
import os
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
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

def perform_grid_search(X_train, y_train):
    parameter_grid = {
        'linear': {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        'rbf': {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]},
        'poly': {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': [0.01, 0.1, 1]},
        'sigmoid': {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    }

    results = {}
    for kernel, grid in parameter_grid.items():
        print(f"Performing grid search for {kernel} kernel")
        clf = GridSearchCV(svm.SVC(class_weight='balanced', decision_function_shape='ovr'), grid, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        results[kernel] = clf
        print(f"Best parameters for {kernel}: {clf.best_params_}")
        print(f"Best cross-validation score: {clf.best_score_:.2f}")

    return results

def train_and_evaluate(data):
    X = data.drop('label', axis=1)
    y = data['label']
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = perform_grid_search(X_train, y_train)

    # Evaluate each model
    performance_metrics = {}
    for kernel, model in results.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        performance_metrics[kernel] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm
        }
        print(f"Results for {kernel}:")
        print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")
        print(f"Confusion Matrix:\n{cm}")

    return performance_metrics

# Path definition
real_data_path = 'GazeTracking/gaze_data/real_data'
synthetic_data_path = 'GazeTracking/gaze_data/synthetic_data'

# File collection and data processing
real_files = glob.glob(os.path.join(real_data_path, "*.csv"))
synthetic_files = glob.glob(os.path.join(synthetic_data_path, "*.csv"))

if real_files:
    print("Processing real data...")
    real_data = process_files(real_files)
    performance_metrics_real = train_and_evaluate(real_data)

if synthetic_files:
    print("Processing synthetic data...")
    synthetic_data = process_files(synthetic_files)
    performance_metrics_synthetic = train_and_evaluate(synthetic_data)

if real_files and synthetic_files:
    print("Processing combined data...")
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
    performance_metrics_combined = train_and_evaluate(combined_data)

# Optionally, save the DataFrame to a CSV file
output_dir = '/mnt/data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df_performance = pd.DataFrame(performance_metrics_real)  # Change as needed or combine results
df_performance.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
