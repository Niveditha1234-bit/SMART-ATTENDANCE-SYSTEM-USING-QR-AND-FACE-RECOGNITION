import cv2
import os

# Path to the dataset folder where images will be stored
dataset_path = r'C:\Clubs\Techsaksham\qr_attendance_system\dataset'

# Get the person ID (this can be their name or any identifier like USN)
person_id = input("Enter person ID (e.g., 1, 2, 3): ")


# Create a subfolder for the person if it doesn't exist
person_folder = os.path.join(dataset_path, person_id)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(r'C:\Clubs\Techsaksham\qr_attendance_system\haarcascade_frontalface_default.xml')

count = 0
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Save the face image
        face_img = gray[y:y + h, x:x + w]
        face_filename = os.path.join(person_folder, f"img{count + 1}.jpg")
        cv2.imwrite(face_filename, face_img)
        count += 1
        print(f"Captured {count} face images...")
    
    # Display the captured image with a rectangle around the face
    cv2.imshow("Capture Faces", frame)
    
    # Break the loop after capturing 30 images (or press 'q' to exit)
    if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
