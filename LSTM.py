import glob
import pandas as pd
import os
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.layers import Bidirectional, Dropout
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from keras.regularizers import l2
import numpy as np


# Directory paths to your data
path_to_real_data = r'GazeTracking/gaze_data/real_data/*.csv'
path_to_synthetic_data = r'GazeTracking/gaze_data/synthetic_data/*.csv'

# Function to load and preprocess data from CSV files
def load_and_preprocess_data(paths):
    sequences = []
    labels = []
    
    all_gaze_directions = set()
    for file_path in glob.glob(paths):
        df = pd.read_csv(file_path)
        # Correct "blinki" to "blinking" before collecting unique directions
        df['gaze_direction'] = df['gaze_direction'].replace('blinki', 'blinking')
        all_gaze_directions.update(df['gaze_direction'].dropna().unique())
    
    # Fit the LabelEncoder with all unique gaze directions
    gaze_encoder = LabelEncoder()
    gaze_encoder.fit(list(all_gaze_directions))
    
    for file_path in glob.glob(paths):
        df = pd.read_csv(file_path)
        # Drop rows with any NaN values
        df.dropna(subset=['left_pupil', 'right_pupil', 'gaze_direction'], inplace=True)
        # Correct "blinki" to "blinking"
        df['gaze_direction'] = df['gaze_direction'].replace('blinki', 'blinking')
        
        # Correctly extracting the classification label from the filename
        label = int(os.path.basename(file_path).split('_')[0])
        
        # Encode gaze direction
        df['gaze_direction'] = gaze_encoder.transform(df['gaze_direction'])
        
        # Extract and convert coordinates
        # Make sure to correctly parse the tuples
        df[['left_pupil_x', 'left_pupil_y']] = df['left_pupil'].str.extract(r'\((.*), (.*)\)').astype(float)
        df[['right_pupil_x', 'right_pupil_y']] = df['right_pupil'].str.extract(r'\((.*), (.*)\)').astype(float)
        sequence = df[['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'gaze_direction']].values
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

# Load and preprocess data
real_sequences, real_labels = load_and_preprocess_data(path_to_real_data)
synthetic_sequences, synthetic_labels = load_and_preprocess_data(path_to_synthetic_data)
print(f"Number of real sequences: {len(real_sequences)}")
print(f"Number of synthetic sequences: {len(synthetic_sequences)}")
# Combine real and synthetic data
all_sequences =  synthetic_sequences + real_sequences 
all_labels =  synthetic_labels + real_labels 
print(f"Total number of sequences: {len(all_sequences)}")

X_padded = pad_sequences(all_sequences, padding='post', dtype='float32')

# Convert labels to categorical
y_categorical = to_categorical(all_labels)

data = X_padded
labels = y_categorical
num_folds = 3
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

accuracy_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

loss_per_fold = []
fold_no = 1
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)
print(f"Number of training sequences: {len(X_train)}")
print(f"Number of testing sequences: {len(X_test)}")
# Define the LSTM model
for train, test in kfold.split(data, labels):

    # Define the model architecture inside the loop
    model = Sequential([
        Input(shape=(data.shape[1], data.shape[2])),
        Masking(mask_value=0.),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.5),
        Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.5),
        Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),
        Dense(labels.shape[1], activation='softmax')
    ])

# Compile the model
    optimizer = Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')


    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=10,  # How many epochs to wait after last time val_loss improved
        verbose=1,
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )
# Train the model
    history = model.fit(data[train], labels[train],
                            batch_size=32,
                            epochs=100,
                            verbose=1,
                            validation_split=0.3)  # You might adjust validation_split if needed

   
    num_validation_sequences = int(len(X_train) * 0.2)
    print(f"Number of validation sequences: {num_validation_sequences}")
    # After training, predict classes on the test set
    predictions = model.predict(data[test])
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels[test], axis=1)

# Define gaze_directions for classification report; adjust according to your actual class names
# Assuming y_train is already converted to categorical with to_categorical
    num_classes = y_train.shape[1]  # This will give you the number of classes

# Now, dynamically set your gaze_directions based on the number of classes
# Here, I'm assuming the classes are sequentially labeled from 0, 1, 2, ...
# Adjust the names based on your actual class names and order
    class_names = ['left', 'right', 'center', 'blinking']  # Example class names
    gaze_directions = class_names[:num_classes]  # Adjust this to match your actual classes


    # Generate generalization metrics
    scores = model.evaluate(data[test], labels[test], verbose=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    loss_per_fold.append(scores[0])
    print(f'Score for fold {fold_no}: Accuracy of {accuracy*100}%, Precision of {precision*100}%, Recall of {recall*100}%, F1 Score of {f1*100}%')
   
    accuracy_per_fold.append(accuracy * 100)
    precision_per_fold.append(precision * 100)
    recall_per_fold.append(recall * 100)
    f1_per_fold.append(f1 * 100)

    fold_no += 1


# Make sure this list matches the classes predicted by the model
# For instance, if your model predicts 3 classes, ensure gaze_directions has 3 names

# Now you can call classification_report without causing a mismatch error
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for label imbalance
recall = recall_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for label imbalance
f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for label imbalance


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(accuracy_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]} - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold} - F1: {f1_per_fold}%')
print('------------------------------------------------------------------------')
print(f'Average scores for all folds:')
print(f'Accuracy: {np.mean(accuracy_per_fold)} +/- {np.std(accuracy_per_fold)}')
print(f'Precision: {np.mean(precision_per_fold)} +/- {np.std(precision_per_fold)}')
print(f'Recall: {np.mean(recall_per_fold)} +/- {np.std(recall_per_fold)}')
print(f'F1 Score: {np.mean(f1_per_fold)} +/- {np.std(f1_per_fold)}')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")