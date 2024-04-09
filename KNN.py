import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_paths):
    data_frames = []
    for file_path in file_paths:
        # Use os.path.basename to ensure we're dealing with just the filename
        filename = os.path.basename(file_path)
        try:
            classification = int(filename.split('_')[0])
        except ValueError as e:
            print(f"Error processing filename: {filename}. Error: {e}")
            continue

        df = pd.read_csv(file_path)
        df['classification'] = classification
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    # Drop rows with any NaN values
    combined_df.dropna(inplace=True)   
    return combined_df

# Paths to your data
path_to_real_data = r'GazeTracking/gaze_data/real_data/*.csv'
path_to_synthetic_data = r'GazeTracking/gaze_data/synthetic_data/*.csv'

# Load real and synthetic data
real_data_files = glob.glob(path_to_real_data)
synthetic_data_files = glob.glob(path_to_synthetic_data)

# Combine both real and synthetic datasets
all_files = real_data_files + synthetic_data_files  

# Load and combine data
data = load_data(all_files)
print(data)
# Preprocess data: Convert pupil coordinates from strings to tuples and then to separate columns
data[['left_pupil_x', 'left_pupil_y']] = data['left_pupil'].str.extract(r'\((.*), (.*)\)').astype(float)
data[['right_pupil_x', 'right_pupil_y']] = data['right_pupil'].str.extract(r'\((.*), (.*)\)').astype(float)
print(data)

data['gaze_direction'] = LabelEncoder().fit_transform(data['gaze_direction'])

# Define features and label
X = data[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'gaze_direction']]
y = data['classification']
print("where am i", X)
print("halloo", y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Predict on test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Sensitivity (Recall): {recall}")
print(f"F1 Score: {f1}")