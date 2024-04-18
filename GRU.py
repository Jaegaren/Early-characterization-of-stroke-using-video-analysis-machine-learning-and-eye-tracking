import pandas as pd
import numpy as np
import os
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Input
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.regularizers import L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Directory paths to your data
real_data_path = "GazeTracking/gaze_data/real_data"
synthetic_data_path = "GazeTracking/gaze_data/synthetic_data"

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
            
# Loading and preprocessing the data by calculating difference in coordinates for each frame
def load_and_preprocess_data(folder_path):
    differences = []  # This will store the coordinate differences for each file
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Extract label from the first character of the file name
            label = int(file_name[0])

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, usecols=['left_pupil', 'right_pupil'])
            
            # Drop rows with missing values
            df = df.dropna()

            # Calculate coordinate differences
            file_differences = []
            previous_row = None
            for _, row in df.iterrows():
                current_row = {
                    'left_pupil': eval(row['left_pupil']),
                    'right_pupil': eval(row['right_pupil'])
                }

                if previous_row is not None:
                    # Calculate differences for left pupil
                    left_diff = (current_row['left_pupil'][0] - previous_row['left_pupil'][0],
                                 current_row['left_pupil'][1] - previous_row['left_pupil'][1])

                    # Calculate differences for right pupil
                    right_diff = (current_row['right_pupil'][0] - previous_row['right_pupil'][0],
                                  current_row['right_pupil'][1] - previous_row['right_pupil'][1])

                    file_differences.append(left_diff + right_diff)

                # Update the previous row
                previous_row = current_row

            differences.append(file_differences)
            labels.append(label)

    return differences, labels

# Determining maximum sequence length
def max_length(real_data, synthetic_data):
    max_len_real = max(len(seq) for seq in real_data)
    max_len_synth = max(len(seq) for seq in synthetic_data)
    return max(max_len_real, max_len_synth)

# Padding using zero-coordinates
def pad_sequence_add(data):
    #max_len = max(max_length(X_train, X_test), max_length(X_train2, X_test2))
    max_len = max_length(real_data, synthetic_data)
    padded_data = []
    for seq in data:
        seq_len = len(seq)
        if seq_len < max_len:
            # Calculate the number of timesteps needed to add to pad the sequence
            padding = [(50,50,50,50)] * (max_len - seq_len)
            # Extend the sequence with the padding
            seq.extend(padding)
        # Append the padded sequence to the list
        padded_data.append(seq)
    # Convert the list of sequences to a NumPy array and return
    return np.array(padded_data, dtype = 'float32')

# Load and preprocess data
real_data, real_labels = load_and_preprocess_data(real_data_path)
synthetic_data, synthetic_labels = load_and_preprocess_data(synthetic_data_path)

# Pad data
real_data_padded = pad_sequence_add(real_data)
synthetic_data_padded = pad_sequence_add(synthetic_data)

# Concatenate data
all_data = np.concatenate((real_data_padded, synthetic_data_padded), axis=0)
all_labels = np.concatenate((real_labels, synthetic_labels), axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.3, random_state = 41)

early_stopping = CustomEarlyStopping(patience=10, verbose=1,)
reg = L1L2(l1=0.005, l2=0.005)

# Transform labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential([
    Input(shape = (max_length(real_data, synthetic_data), 4)),
    Masking(mask_value = 50),
    GRU(256, return_sequences = True, kernel_regularizer = reg),
    Dropout(0.5),
    GRU(128, return_sequences = True, kernel_regularizer=reg),
    Dropout(0.5),
    GRU(64, return_sequences = True, kernel_regularizer=reg),
    Dropout(0.5),
    GRU(32),
    Dense(3, activation='softmax')
])

model.compile(optimizer = Adam(learning_rate=0.0008),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy', 'Precision', 'Recall'])

historall_labels = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs = 100,
                    batch_size = 32,
                    callbacks = early_stopping)

training_loss = historall_labels.history['loss']
validation_loss = historall_labels.history['val_loss']
training_accuracy = historall_labels.history['accuracy']
validation_accuracy = historall_labels.history['val_accuracy']
# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

scores = model.evaluate(X_test, y_test, verbose=0)
print(f'Score: {model.metrics_names} of {scores}')

predictions = model.predict(X_test) 
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(10, 7))

# Convert the counts to percentages for annotations
cm_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
cm_labels = [f"{v1}\n{v2}" for v1, v2 in zip(cm.flatten(), cm_percentages)]
cm_labels = np.asarray(cm_labels).reshape(cm.shape)

# Create the heatmap using seaborn
sns.heatmap(cm, annot=cm_labels, fmt="", cmap='Blues', ax=ax)

# Set labels and title
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix with Percentages and Counts')

# Set ticks
tick_marks = np.arange(len(cm)) + 0.5
plt.xticks(tick_marks, range(len(cm)))
plt.yticks(tick_marks, range(len(cm)), rotation=0)

# Show the plot
plt.show()


print("Validation scores:")
print(f'Loss: {round(scores[0], 2)}')
print(f'Precision: {round(scores[1] * 100, 2)}%')
print(f'Recall: {round(scores[2] * 100, 2)}%')
print(f'Accuracy: {round(scores[3] * 100, 2)}%')

# Visualizing the loss as the network learns
plt.figure(figsize=(10, 6))
plt.plot(epoch_count, training_loss, 'r--', label='Training Loss')
plt.plot(epoch_count, validation_loss, 'b-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
