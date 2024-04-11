import cv2
from gaze_tracking import GazeTracking
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import backend as K


# Initialize GazeTracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

recording = False
data = []

def preprocess_data(sequence):
    X_padded = pad_sequences([sequence], padding='post', dtype='float32')
    return X_padded

def predict_classification(model, sequence):
    # Assuming model's output is categorical and directly corresponds to 0, 1, 2 classification
    prediction = model.predict(sequence)
    classification = np.argmax(prediction, axis=1)
    return classification[0]

# Clear session
K.clear_session()

# Load the LSTM model
model = load_model('final_lstm_model.keras')

while True:
    # Read frame
    _, frame = webcam.read()
    gaze.refresh(frame)
    
    frame = gaze.annotated_frame()
    text = "Press space to start/stop recording"

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display frame
    cv2.imshow("Gaze Tracking", frame)

    key = cv2.waitKey(1)
    if key == 32: # Space bar
        if recording:
            # Stop recording
            recording = False
            
            # Convert data to DataFrame and save to CSV
            columns = ['left_pupil', 'right_pupil', 'gaze_direction']
            df = pd.DataFrame(data, columns=columns)
            csv_filename = 'gaze_data.csv'
            df.to_csv(csv_filename, index=False)
            
            # Preprocess the data for the LSTM model
            sequence = [[(row[0][0]+row[1][0])/2, (row[0][1]+row[1][1])/2] for row in data if None not in row]  # Using average pupil positions
            X_processed = preprocess_data(sequence)
            
            # Predict classification
            classification = predict_classification(model, X_processed)
            print(f"Classification: {classification}")
            
            data = []  # Reset data
            
            print("Recording stopped and data saved to", csv_filename)
        else:
            # Start recording
            recording = True
            print("Recording started...")
    elif key == 27: # ESC key
        break
    
    if recording:
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        direction = "center"  # Default value
        if gaze.is_blinking():
            direction = "blinking"
        elif gaze.is_right():
            direction = "right"
        elif gaze.is_left():
            direction = "left"
        
        # Append current frame data
        data.append([left_pupil, right_pupil, direction])

# Cleanup
webcam.release()
cv2.destroyAllWindows()
