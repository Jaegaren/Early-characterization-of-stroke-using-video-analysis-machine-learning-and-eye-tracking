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
        df = pd.read_csv(filename)
        df[['left_pupil_x', 'left_pupil_y']] = df['left_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        df[['right_pupil_x', 'right_pupil_y']] = df['right_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)

        df_aggregated = df.agg({
            'left_pupil_x': ['mean', 'std', 'max', 'min'],
            'left_pupil_y': ['mean', 'std', 'max', 'min'],
            'right_pupil_x': ['mean', 'std', 'max', 'min'],
            'right_pupil_y': ['mean', 'std', 'max', 'min']
        }).transpose()

        df_aggregated = df_aggregated.unstack().to_frame().transpose()
        df_aggregated.columns = ['_'.join(map(str, col)) for col in df_aggregated.columns.values]
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

def calculate_specificity(conf_matrix, class_index):
    total_true = np.sum(conf_matrix)
    TP = conf_matrix[class_index, class_index]
    FP = np.sum(conf_matrix[:, class_index]) - TP
    FN = np.sum(conf_matrix[class_index, :]) - TP
    TN = total_true - TP - FP - FN
    return TN / (TN + FP) if (TN + FP) > 0 else 0

def calculate_average_specificity(conf_matrix):
    num_classes = conf_matrix.shape[0]
    specificities = [calculate_specificity(conf_matrix, i) for i in range(num_classes)]
    return np.mean(specificities)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    results = perform_grid_search(X_train, y_train)

    performance_metrics = {}
    for kernel, model in results.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        specificity = calculate_average_specificity(cm)

        performance_metrics[kernel] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity,
            'Confusion Matrix': cm
        }
        print(f"Results for {kernel}:")
        print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}, Specificity: {specificity:.2%}")
        print(f"Confusion Matrix:\n{cm}")

    return performance_metrics

# Path definition
real_data_path = 'GazeTracking/gaze_data/real_data'
synthetic_data_path = 'GazeTracking/gaze_data/synthetic_data'

# File collection and data processing
real_files = glob.glob(os.path.join(real_data_path, "*.csv"))
synthetic_files = glob.glob(os.path.join(synthetic_data_path, "*.csv"))

# Processing files
if real_files:
    print("Processing real data...")
    real_data = process_files(real_files)

if synthetic_files:
    print("Processing synthetic data...")
    synthetic_data = process_files(synthetic_files)

# Scenario 1: Train on real data, test on real data
print("Scenario 1: Train on real data, test on real data")
X_real = real_data.drop('label', axis=1)
y_real = real_data['label']
X_real = SimpleImputer(strategy='mean').fit_transform(X_real)
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
performance_metrics_real = train_and_evaluate(X_train_real, y_train_real, X_test_real, y_test_real)

# Scenario 2: Train on synthetic data, test on real data
print("Scenario 2: Train on synthetic data, test on real data")
X_synthetic = synthetic_data.drop('label', axis=1)
y_synthetic = synthetic_data['label']
X_synthetic = SimpleImputer(strategy='mean').fit_transform(X_synthetic)
performance_metrics_synthetic = train_and_evaluate(X_synthetic, y_synthetic, X_test_real, y_test_real)

# Scenario 3: Train on combined data, test on combined data
print("Scenario 3: Train on combined data, test on combined data")
combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
X_combined = combined_data.drop('label', axis=1)
y_combined = combined_data['label']
X_combined = SimpleImputer(strategy='mean').fit_transform(X_combined)
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)
performance_metrics_combined = train_and_evaluate(X_train_combined, y_train_combined, X_test_combined, y_test_combined)

# Save performance metrics
output_dir = '/mnt/data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save each scenario's performance metrics separately
df_performance_real = pd.DataFrame(performance_metrics_real)
df_performance_real.to_csv(os.path.join(output_dir, 'performance_metrics_real.csv'), index=False)

df_performance_synthetic = pd.DataFrame(performance_metrics_synthetic)
df_performance_synthetic.to_csv(os.path.join(output_dir, 'performance_metrics_synthetic.csv'), index=False)

df_performance_combined = pd.DataFrame(performance_metrics_combined)
df_performance_combined.to_csv(os.path.join(output_dir, 'performance_metrics_combined.csv'), index=False)

