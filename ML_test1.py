import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob
import os
import numpy as np

# Load and combine CSV files
path = r'GazeTracking/gaze_data'  # Use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    # Extract label from filename
    label = os.path.basename(filename).split('_')[0]  # This assumes the format "Label_Filename.csv"
    df = pd.read_csv(filename, index_col=None, header=0)
    df['label'] = label  # Add the extracted label to the DataFrame
    
    # Preprocess pupil coordinates: Convert from string to separate numeric columns
    df['left_pupil_x'] = df['left_pupil'].str.extract(r'\((.*),')[0].astype(float)
    df['left_pupil_y'] = df['left_pupil'].str.extract(r', (.*)\)')[0].astype(float)
    df['right_pupil_x'] = df['right_pupil'].str.extract(r'\((.*),')[0].astype(float)
    df['right_pupil_y'] = df['right_pupil'].str.extract(r', (.*)\)')[0].astype(float)
    
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

# Convert gaze direction to a numeric label if it's not already
frame['gaze_direction_label'] = frame['gaze_direction'].factorize()[0]

# Selecting the new features for the model
X = frame[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'gaze_direction_label']]
y = frame['label']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', pos_label='Stroke')
recall = recall_score(y_test, y_pred, average='binary', pos_label='Stroke')
f1 = f1_score(y_test, y_pred, average='binary', pos_label='Stroke')

# Calculate specificity safely
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['Not_stroke', 'Stroke']).ravel()
if (tn + fp) > 0:
    specificity = tn / (tn + fp)
else:
    specificity = np.nan  # Or consider another approach to handle this case

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Sensitivity (Recall): {recall}")
print(f"Specificity: {specificity if not np.isnan(specificity) else 'Undefined'}")
print(f"F1 Score: {f1}")
