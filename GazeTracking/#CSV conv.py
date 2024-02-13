import os
import cv2
from gaze_tracking import GazeTracking
import pandas as pd

# Initialize gaze tracking
gaze = GazeTracking()

# Define your directory structure
base_dir = r"C:\Users\est02\Downloads\Videor"  # Adjust path as needed, using raw string to avoid unicode error

# Categories and their subdirectories based on your file structure
categories = {
    "Stroke": ["Finger-test/Eye-deviation/Start left", "Finger-test/Eye-deviation/Start mid", "Finger-test/Eye-deviation/Start right"],
    "Frisk data": ["Med penna/Finger-test", "Utan penna/Fingertest"]
}

# Placeholder for extracted data
data = []

# Process videos and extract features
for category, paths in categories.items():
    for sub_path in paths:
        full_path = os.path.join(base_dir, category, sub_path)
        label = sub_path.replace('/', '_')  # Creating a label from the sub-path, replacing slashes with underscores

        # Iterate through videos in each sub-directory
        for video_name in os.listdir(full_path):
            # Skip .XML files, process all other files
            if video_name.lower().endswith('.xml'):
                continue

            print(f"Processing video: {video_name}")
            video_path = os.path.join(full_path, video_name)
            video = cv2.VideoCapture(video_path)

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                gaze.refresh(frame)

                # Extract features
                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()
                gaze_direction = 'center'  # Default value
                if gaze.is_blinking():
                    gaze_direction = 'blinking'
                elif gaze.is_right():
                    gaze_direction = 'right'
                elif gaze.is_left():
                    gaze_direction = 'left'
                
                # Append data
                data.append([video_name, left_pupil, right_pupil, gaze_direction, label])

            video.release()


# Convert list to DataFrame
df = pd.DataFrame(data, columns=['video_name', 'left_pupil', 'right_pupil', 'gaze_direction', 'label'])

# Save DataFrame to a CSV file for further processing
df.to_csv('gaze_data.csv', index=False)
