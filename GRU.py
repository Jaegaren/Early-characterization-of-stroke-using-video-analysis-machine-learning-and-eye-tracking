import pandas as pd
import numpy as np
import os
import tensorflow
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import Input
from keras.layers import Masking
from keras.metrics import Precision, Recall, AUC, F1Score
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# Loading and preprocessing the data
def load_and_preprocess_data(folder_path):
    data = []
    labels = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Extract label from the first character of the file name
            label = int(file_name[0])

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, usecols=['left_pupil', 'right_pupil'])
            
            # Drop rows with missing values for real data (only real_data can have missing values)
            if 'real_data' in folder_path:
                df = df.dropna()
                
            # Extract coordinates
            coordinates = []
            for _, row in df.iterrows():
                left_pupil = eval(row['left_pupil'])
                right_pupil = eval(row['right_pupil'])
                coordinates.append(left_pupil + right_pupil)  # Concatenate coordinates
            
            data.append(coordinates)
            labels.append(label)
            
    return data, labels


'''
# Padding using the last know coordinates
def pad_sequence_add(data):
    # Determine max sequence length
    max_len = max(len(seq) for seq in real_data)
    
    # Pad sequences with the last known coordinate
    padded_data = []
    for seq in data:
        seq_len = len(seq)
        if seq_len < max_len:
            last_coords = seq[-1]
            padding = [last_coords] * (max_len - seq_len)
            seq.extend(0)
        padded_data.append(seq)
    return np.array(padded_data)
'''


# Determining maximum sequence length
def max_length(real_data, synthetic_data):
    max_len_real = max(len(seq) for seq in real_data)
    max_len_synth = max(len(seq) for seq in real_data)
    return max(max_len_real, max_len_synth)


# Determining minimum sequence length
def min_length(real_data, synthetic_data):
    max_len_real = min(len(seq) for seq in real_data)
    max_len_synth = min(len(seq) for seq in real_data)
    return min(max_len_real, max_len_synth)


# Padding using zero-coordinates
def pad_sequence_add(data):
    max_len = max_length(real_data, synthetic_data)
    padded_data = []
    for seq in data:
        seq_len = len(seq)
        if seq_len < max_len:
            # Calculate the number of timesteps needed to add to pad the sequence
            padding = [(0,0,0,0)] * (max_len - seq_len)
            # Extend the sequence with the padding
            seq.extend(padding)
        # Append the padded sequence to the list
        padded_data.append(seq)
    # Convert the list of sequences to a NumPy array and return
    return np.array(padded_data)


# Adapting all sequences to equal length of the shortest sequence
def pad_sequence_remove(data):
    min_len = min_length(real_data, synthetic_data)
    processed_data = []
    for seq in data:
        seq_len = len(seq)
        if seq_len > min_len:
            # Trim the sequence if it's longer than min_len
            processed_seq = seq[:min_len]
        elif seq_len < min_len:
            # Extend the sequence with the last element if it's shorter than min_len
            last_coords = seq[-1]
            padding = [last_coords] * (min_len - seq_len)
            processed_seq = seq + padding
        else:
            # If the sequence is already the correct length, use it as is
            processed_seq = seq
        # Append the processed sequence to the list
        processed_data.append(processed_seq)
    # Convert the list of sequences to a NumPy array and return
    return np.array(processed_data)

# Load and preprocess data
real_data_path = "GazeTracking/gaze_data/real_data"
synthetic_data_path = "GazeTracking/gaze_data/synthetic_data"
real_data, real_labels = load_and_preprocess_data(real_data_path)
synthetic_data, synthetic_labels = load_and_preprocess_data(synthetic_data_path)

# Pad data
real_data_padded = pad_sequence_add(real_data)
synthetic_data_padded = pad_sequence_add(synthetic_data)

# Concatenate data
all_data = np.concatenate((real_data_padded, synthetic_data_padded), axis=0)
all_labels = np.concatenate((real_labels, synthetic_labels), axis=0)


# Convert labels to categorical
#all_labels = to_categorical(all_labels)

# Creating early stopping variable
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor
    patience=10,                 # Number of epochs with no improvement after which training will be stopped
    verbose=5,                  # To log when training is stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
)

# Define the number of splits
num_folds = 5

# Define K-fold 
kfold = KFold(n_splits = num_folds, shuffle=True, random_state = 42)

# Define stratified k-fold 
skf = StratifiedKFold(n_splits = num_folds, shuffle=True, random_state = 42)

# Placeholder for fold performance
scores_per_fold = []

fold_no = 1
for train, test in skf.split(all_data, all_labels):
    y_train = to_categorical(all_labels[train])
    y_test = to_categorical(all_labels[test])

    model = Sequential([
        Input(shape = (max_length(real_data, synthetic_data), 4)),
        Masking(mask_value = (0,0,0,0)), 
        GRU(128, return_sequences=True),
        GRU(64, return_sequences=True),
        GRU(32),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer = Adam(learning_rate=0.001),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy', 'Precision', 'Recall'])

    print(f'Training for fold {fold_no}...')
    
    historall_labels = model.fit(all_data[train], y_train,
                        validation_data=(all_data[test], y_test),
                        epochs = 100,
                        batch_size = 32,
                        callbacks = early_stopping)
    
    scores = model.evaluate(all_data[test], y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names} of {scores}')
    scores_per_fold.append(scores)
    
    fold_no += 1

# Calculate and print the average scores across all folds
average_scores = np.mean(scores_per_fold, axis=0)
print(f'Average scores across all folds: {average_scores}')


'''
# Split data
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

X_train = tensorflow.convert_to_tensor(X_train, dtype=tensorflow.float32)
X_test = tensorflow.convert_to_tensor(X_test, dtype=tensorflow.float32)

#y_train = to_categorical(y_train, num_classes=3)
#y_test = to_categorical(y_test, num_classes=3)

model = Sequential([
    Input(shape=(None, 4)),
    Masking(mask_value = (0,0,0,0)),
    GRU(32),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'F1Score'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size = 32, callbacks=[early_stopping])
'''


