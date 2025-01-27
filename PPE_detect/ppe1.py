from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import threading  # For playing and stopping alert sound
import os
from playsound import playsound  # Import for playing sound

# Function to play alert sound in a loop
def play_alert():
    while alert_playing:
        playsound("alert.mp3", block=False)

# Choose input source: camera, video, or image
choice = input("Enter '1' for Camera, '2' for Video, or '3' for Image: ")

if choice == '1':
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height
elif choice == '2':
    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit()
    cap = cv2.VideoCapture(video_path)  # Open video file
elif choice == '3':
    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        exit()
    img = cv2.imread(image_path)  # Read the image
else:
    print("Invalid choice! Exiting...")
    exit()

# Load the YOLO model
model = YOLO("model.pt")

# Define class names
classNames = {
    0: 'Glass', 1: 'Gloves', 2: 'Goggles', 3: 'Helmet', 4: 'No-Helmet',
    5: 'No-Vest', 6: 'Person', 7: 'Safety-Boot', 8: 'Safety-Vest', 9: 'Vest',
    10: 'helmet', 11: 'no helmet', 12: 'no vest', 13: 'no_helmet', 14: 'no_vest',
    15: 'protective_suit', 16: 'vest', 17: 'worker'
}

prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initialize fps variable
alert_playing = False  # Flag to check if the alert sound is playing
alert_thread = None  # Thread for playing the alert sound

# Handle image file detection
if choice == '3':
    results = model(img, stream=True)
    alert_triggered = False  # Flag to check if alert needs to be triggered

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Display class name and confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Trigger alert if confidence >= 0.2
            if conf >= 0.2:
                alert_triggered = True

    # Handle alert sound
    if alert_triggered and not alert_playing:
        alert_playing = True
        alert_thread = threading.Thread(target=play_alert)
        alert_thread.start()
    elif not alert_triggered and alert_playing:
        alert_playing = False
        if alert_thread:
            alert_thread.join()  # Wait for the thread to finish

    # Display image with detection
    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close all OpenCV windows
    exit()

# Handle video or camera detection
if not cap.isOpened():
    print("Error: Camera/Video not found or could not be opened.")
    exit()

try:
    while True:
        success, img = cap.read()

        if not success:
            print("Failed to capture image or end of video.")
            break

        results = model(img, stream=True)
        alert_triggered = False  # Flag to check if alert needs to be triggered

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                # Display class name and confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Trigger alert if confidence >= 0.2
                if conf >= 0.2:
                    alert_triggered = True

        # Handle alert sound
        if alert_triggered and not alert_playing:
            alert_playing = True
            alert_thread = threading.Thread(target=play_alert)
            alert_thread.start()
        elif not alert_triggered and alert_playing:
            alert_playing = False
            if alert_thread:
                alert_thread.join()  # Wait for the thread to finish

        # Calculate FPS
        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            fps = int(fps)  # Convert to integer
        prev_frame_time = new_frame_time

        # Display FPS on the image
        cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

        print(f'FPS: {fps}')
        
        cv2.imshow("Video/Camera Detection", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break
finally:
    alert_playing = False  # Stop the alert sound
    if alert_thread:
        alert_thread.join()  # Ensure the alert sound thread stops
    cap.release()  # Release the camera or video when done
    cv2.destroyAllWindows()  # Close all OpenCV windows
