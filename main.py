import cv2
import os
import numpy as np
import csv
from datetime import datetime
from pyzbar.pyzbar import decode

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(r'C:\Clubs\Techsaksham\qr_attendance_system\haarcascade_frontalface_default.xml')

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset folder where images for training are stored
dataset_path = r'C:\Clubs\Techsaksham\qr_attendance_system\dataset'

# Path to the attendance log file
attendance_log_path = r'C:\Clubs\Techsaksham\qr_attendance_system\attendance_log.csv'

# Function to load the dataset and train the recognizer
def train_recognizer():
    faces = []
    labels = []
    label_dict = {}  # Dictionary to map labels to names
    
    # Loop through the dataset folder and collect images and labels
    for label_folder in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_folder)
        
        if os.path.isdir(label_folder_path):
            label = int(label_folder)  # Assuming the folder name is the label
            label_dict[label] = label_folder  # Map label to name
            
            # Collect images in each folder
            for image_name in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_name)
                
                # Load the image, convert to grayscale, and append to faces list
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                faces.append(gray)
                labels.append(label)
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    
    return label_dict

# Train the recognizer and get the label-to-name dictionary
label_dict = train_recognizer()

# Function to log attendance in CSV
def log_attendance(name, confidence, filename):
    with open(attendance_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the name, confidence score, image filename, and current time (attendance mark)
        writer.writerow([name, confidence, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Create CSV file if it doesn't exist
if not os.path.isfile(attendance_log_path):
    with open(attendance_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Confidence", "Image Filename", "Timestamp"])

# Function to scan QR Code
def scan_qr_code():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Decode QR code in the frame
        qr_codes = decode(frame)
        
        for qr_code in qr_codes:
            # If a QR code is detected, extract the data
            qr_data = qr_code.data.decode('utf-8')
            print(f"QR Code Scanned: {qr_data}")
            
            # Close QR scanner window and return the scanned data
            cap.release()
            cv2.destroyAllWindows()
            return qr_data
        
        # Show the video frame
        cv2.imshow("Scan QR Code", frame)
        
        # Exit the QR scanner on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Flag to check if the attendance is already marked
attendance_marked = False

# Step 1: Scan the QR code
qr_data = scan_qr_code()

if qr_data:
    print(f"QR Code Scanned, proceeding to face recognition for: {qr_data}")
    
    # Step 2: Initialize the camera for face recognition
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If a face is detected
        if len(faces) > 0:
            # Take the first detected face (if there are multiple faces, we only process the first one)
            (x, y, w, h) = faces[0]

            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Region of interest for face recognition
            roi_gray = gray[y:y + h, x:x + w]

            # Recognize the face
            label, confidence = recognizer.predict(roi_gray)

            # Get the recognized name from the label dictionary
            name = label_dict.get(label, "Unknown")
            text = f"Name: {name}, Confidence: {confidence:.2f}"

            # Display details on the screen
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Step 3: If the person is recognized and attendance is not marked yet
            if name != "Unknown" and not attendance_marked:
                # Log attendance for the recognized person
                # Save the image of the recognized face
                recognized_face_filename = os.path.join('recognized_faces', f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                os.makedirs('recognized_faces', exist_ok=True)
                cv2.imwrite(recognized_face_filename, roi_gray)
                print(f"Attendance marked for {name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Log the attendance with additional details
                log_attendance(name, confidence, recognized_face_filename)

                # Mark the attendance as done
                attendance_marked = True

            # Step 4: Break the loop after the face is recognized and attendance is marked
            if attendance_marked:
                break

        # Show the image with face rectangles and details
        cv2.imshow('Face Recognition & Attendance', img)

        # Exit the loop when the user presses the "q" key
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# End of the code
