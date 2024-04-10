import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import os

print("Starting script...")

# Load and combine CSV files
path = os.path.join('GazeTracking', 'gaze_data', 'real_data')
all_files = glob.glob(os.path.join(path, "*.csv"))

if not all_files:
    print(f"No CSV files found in directory {path}")
else:
    li = []

    for filename in all_files:
        label = int(os.path.basename(filename).split('_')[0])  # Extract label from filename
        df = pd.read_csv(filename, index_col=None, header=0)
        df['label'] = label
        df[['left_pupil_x', 'left_pupil_y']] = df['left_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        df[['right_pupil_x', 'right_pupil_y']] = df['right_pupil'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
        li.append(df)

    print("Dataframes created, concatenating...")

    # Combine all dataframes
    frame = pd.concat(li, axis=0, ignore_index=True)

    print("Concatenated dataframes, selecting features and labels...")

    # Select features and labels
    X = frame[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]
    y = frame['label']

    print("Selected features and labels, checking for NaN values...")

    # Check for NaN values and fill them if found
    if X.isnull().values.any():
        print(f"NaN values found in the features, filling with mean of the columns")
        X = X.fillna(X.mean())

    print("Splitting dataset into training and test sets...")

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Creating SVM classifier...")

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    print("Training the model...")

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    print("Model trained, predicting the test dataset...")

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, etc.
    print("Calculating metrics...")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Sensitivity (Recall): {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

print("Script completed.")
