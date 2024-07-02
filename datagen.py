import cv2
import os


def extract_frames(video_path, output_folder, interval=3):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = 30

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save frame as an image file
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    video.release()

    print(f"Extracted {saved_count} frames")


# Usage example
video_path = "test.webm"
output_folder = "calibration_data"
extract_frames(video_path, output_folder)
