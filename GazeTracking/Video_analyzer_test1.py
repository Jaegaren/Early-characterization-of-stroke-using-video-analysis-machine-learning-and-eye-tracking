import cv2
from gaze_tracking import GazeTracking

# Initialize the gaze tracking library
gaze = GazeTracking()

# Load the video file instead of the webcam feed
# FOR PYCHARM:
#video_path = 'Videos/C0003.MP4'  # Relative path from the project's root directory
# FOR VSCODE:
video_path = 'GazeTracking/Videos/C0003.MP4'  # Relative path from the project's root directory
video = cv2.VideoCapture(video_path)

while True:
    # Read a new frame from the video
    ret, frame = video.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Analyze the gaze in the current frame
    gaze.refresh(frame)

    # Prepare the frame for showing (optional, if you want to visualize)
    frame = gaze.annotated_frame()
    text = "Gaze Direction: "

    # Determine the gaze direction
    if gaze.is_blinking():
        text += "Blinking"
    elif gaze.is_right():
        text += "Looking right"
    elif gaze.is_left():
        text += "Looking left"
    elif gaze.is_center():
        text += "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    # Print the gaze direction to the terminal
    print(text)

    # Get pupil coordinates (optional, if you need this information)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    print(f"Left pupil: {left_pupil}, Right pupil: {right_pupil}")

    # Resize the frame before showing
    display_size = (1280, 720)  # Set this to your desired display size
    resized_frame = cv2.resize(frame, display_size)

    # Show the frame with annotations (optional, if you want to visualize)
    cv2.imshow("Gaze Tracking", resized_frame)

    # Press 'ESC' to exit the loop (if showing the frame)
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and destroy all OpenCV windows
video.release()
cv2.destroyAllWindows()
