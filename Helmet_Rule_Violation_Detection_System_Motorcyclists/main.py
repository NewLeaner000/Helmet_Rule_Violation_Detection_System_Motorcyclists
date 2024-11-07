import cv2
import math
import numpy as np
from ultralytics import YOLO

# Create a Video Capture Object for a video file
cap = cv2.VideoCapture("./sample/street.mp4")

# Initialize the YOLOv8 Model
model = YOLO('./model/helmet_detection_licenseplate_recognition.pt')

# ClassNames
classNames = ['helmet', 'licenseplate', 'motorcyclist', 'nohelmet']

# Preprocessing function for license plate
def preprocess_license_plate(license_plate_crop):
    # Convert to grayscale
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Histogram equalization to improve contrast
    equalized = cv2.equalizeHist(blurred)

    # Adaptive Thresholding to handle lighting conditions
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Optional: Apply edge detection (Canny) to highlight edges
    edges = cv2.Canny(thresh, 50, 150)

    # Dilation to close gaps in the characters and make them clearer
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Erosion to remove small noise (optional)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Resize frame to improve speed
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = model.predict(frame, conf=0.6)  # Increase confidence threshold to filter detections

    # Loop over each detection
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Set color based on the detected class
            if conf > 0.6:  # Further increase confidence threshold for more stable real-time output
                if currentClass == 'nohelmet':
                    myColor = (0, 0, 255)  # Red for 'nohelmet'
                elif currentClass == 'helmet':
                    myColor = (0, 255, 0)  # Green for 'helmet'
                else:
                    myColor = (255, 0, 0)  # Blue for other classes
                
                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 2)
                label = f'{currentClass} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, myColor, 2)

                # Process the license plate if detected
                if currentClass == 'licenseplate':
                    # Crop the license plate area
                    license_plate_crop = frame[y1:y2, x1:x2]

                    # Preprocess the license plate image
                    processed_license_plate = preprocess_license_plate(license_plate_crop)

                    # Resize the processed license plate to "zoom" in for better visualization
                    zoom_size = (400, 300)  # Define the zoomed size (you can adjust this)
                    license_plate_crop_resized = cv2.resize(license_plate_crop, zoom_size, interpolation=cv2.INTER_LINEAR)
                    processed_license_plate_resized = cv2.resize(processed_license_plate, zoom_size, interpolation=cv2.INTER_LINEAR)


                    # Display the zoomed-in and processed versions
                    cv2.imshow("Zoomed License Plate", license_plate_crop_resized)
                    cv2.imshow("Processed License Plate", processed_license_plate_resized)

    # Display the video frame with bounding boxes
    cv2.imshow("Real-Time Video Processing", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
