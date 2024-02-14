import os
import cv2
from gaze_tracking import GazeTracking
import pandas as pd

# Initialize gaze tracking
gaze = GazeTracking()

# Define your directory structure
# FOR EDDIE'S COMPUTER:
#base_dir = r"C:\Users\est02\OneDrive - Chalmers\Kandidat_vids\Videor"  # Adjust path as needed, using a raw string
# FOR ROBIN'S COMPUTER:
base_dir = r"C:\Users\Robin Khatiri\OneDrive\Desktop\shid\Videor"  # Adjust path as needed, using a raw string

# Custom labels based on directory paths
custom_labels = {
    "Stroke/Finger-test/Eye-deviation/Start left": "Stroke_LeftDeviation",
    "Stroke/Finger-test/Eye-deviation/Start mid": "Stroke_MidDeviation",
    "Stroke/Finger-test/Eye-deviation/Start right": "Stroke_RightDeviation",
    "Frisk data/Med penna/Finger-test": "Not_stroke",
    "Frisk data/Utan penna/Fingertest": "Not_stroke"
}

# Placeholder for extracted data
data = []

# Track directories from which a video has already been processed
processed_dirs = set()

# Process videos and extract features
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Check if the directory has been processed
        if root in processed_dirs:
            continue  # Skip remaining files if one has been processed

        if not file.lower().endswith('.xml'):  # Skip .XML files
            video_path = os.path.join(root, file)
            video = cv2.VideoCapture(video_path)

            # Print the name of the video being processed
            print(f"Processing video: {file}")

            # Determine label based on the directory structure
            relative_path = os.path.relpath(root, base_dir)  # Get the relative path from the base directory
            label = "Unknown"  # Default label
            for path, assigned_label in custom_labels.items():
                if relative_path.replace('\\', '/').startswith(path):
                    label = assigned_label
                    break

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
                data.append([file, left_pupil, right_pupil, gaze_direction, label])

            video.release()

            # Mark this directory as processed
            processed_dirs.add(root)

# Convert list to DataFrame
df = pd.DataFrame(data, columns=['video_name', 'left_pupil', 'right_pupil', 'gaze_direction', 'label'])

# Save DataFrame to a CSV file for further processing
df.to_csv('gaze_data.csv', index=False)
