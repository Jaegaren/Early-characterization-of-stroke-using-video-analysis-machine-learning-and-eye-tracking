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
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from keras.layers import LSTM
from keras.layers import Bidirectional, Dropout
from keras.regularizers import l2

class CustomEarlyStopping(Callback):
    def __init__(self, patience=20, verbose=1):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.best_weights = None
        self.best_metric = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_accuracy = logs.get('val_accuracy')

        # Normalize the accuracy: Assuming accuracy ranges from 0 to 1
        normalized_accuracy = 1 - current_accuracy  # Convert to a loss-like metric (lower is better)

        # Calculate the average of normalized accuracy and validation loss
        combined_metric = (current_val_loss + normalized_accuracy) / 2

        # Check if the combined metric improved
        if np.less(combined_metric, self.best_metric):
            self.best_metric = combined_metric
            self.best_weights = self.model.get_weights()
            self.wait = 0
            print(' NEW BEST WEIGHTS ')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                if self.verbose > 0:
                    print(f'Epoch {epoch+1}: early stopping')


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
            
            # Drop rows with missing values 
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
    return np.array(padded_data, dtype = 'float32')


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

# Define the number of splits
num_folds = 3

# Define K-fold 
kfold = KFold(n_splits = num_folds, shuffle=True, random_state = 42)

# Define stratified k-fold 
skf = StratifiedKFold(n_splits = num_folds, shuffle=True, random_state = 42)

# Placeholder for fold performance
scores_per_fold = []

# Define model type
gru = False
lstm = True

# Settig early stopping condition


'''
    early_stopping = EarlyStopping(
        monitor='val_loss',         # Metric to monitor
        patience=10,                 # Number of epochs with no improvement after which training will be stopped
        verbose=1,                  # To log when training is stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    )
'''

# Split data
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.3, random_state = 42)

print('Train set: ', y_train)
print('Test set: ', y_test)

fold_no = 1
if gru:
    # GRU model
    print('-----GRU-----')

    # For-loop for k-folding
    #for train, test in skf.split(all_data, all_labels):
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = Sequential([
        Input(shape = (max_length(real_data, synthetic_data), 4)),
        Masking(mask_value = 0), 
        GRU(128, return_sequences = True),
        GRU(64, return_sequences = True),
        GRU(32),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer = Adam(learning_rate=0.001),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy', 'Precision', 'Recall'])
    print(f'Training for fold {fold_no}...')
    
    historall_labels = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs = 20,
                        batch_size = 32,
                        callbacks = early_stopping)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names} of {scores}')
    scores_per_fold.append(scores)
    
    fold_no += 1

if lstm:

    early_stopping = CustomEarlyStopping(patience=20, verbose=1)

    # LSTM model
    print('-----LSTM-----')
    
    # For-loop for k-folding
    for train, test in skf.split(all_data, all_labels):
        print(f'Training for fold {fold_no}...')        

        y_train = to_categorical(all_labels[train])
        y_test = to_categorical(all_labels[test])
        model = Sequential([
            Input(shape = (max_length(real_data, synthetic_data), 4)),
            Masking(mask_value = (0,0,0,0)), 
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
            Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001))),
            Bidirectional(LSTM(16, kernel_regularizer=l2(0.001))),
            Dense(3, activation='softmax')
            
        ])
        
        model.compile(optimizer = 'adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy', 'Precision', 'Recall'])
        
        
        historall_labels = model.fit(all_data[train], y_train,
                            validation_data=(all_data[test], y_test),
                            epochs = 100,
                            batch_size = 32,
                            callbacks = early_stopping)
        
        scores = model.evaluate(all_data[test], y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names} of {scores}')
        scores_per_fold.append(scores)

        predictions = model.predict(all_data[test])  # Use X_test here
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)  # Use y_test here
        print(f'Prediction fold {fold_no}:  ', y_pred)
        print(f'True labels fold {fold_no}: ', y_true)
        fold_no += 1
        
        

# Calculate and print the average scores across all folds
average_scores = np.mean(scores_per_fold, axis=0)
print("Average validation scores across all folds:")
print(f'Loss: {round(average_scores[0], 2)}')
print(f'Precision: {round(average_scores[1] * 100, 2)}%')
print(f'Recall: {round(average_scores[2] * 100, 2)}%')
print(f'Accuracy: {round(average_scores[3] * 100, 2)}%')




