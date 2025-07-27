SMART ATTENDANCE SYSTEM USING QR AND FACE RECOGNITION

The Smart Attendance System integrates QR code scanning and face recognition to automate attendance tracking in educational institutions and workplaces. Traditional methods like manual attendance or RFID cards are prone to errors and proxies. This system ensures accuracy, efficiency, and security by verifying both a QR code generated per student and face recognition authentication to mark attendance.

This system uses a dual authentication process:
-QR Code-based Check-in – Each student scans a unique QR code generated for the session.
-Face Recognition Verification – The system captures and matches the student’s face with pre-registered images to validate their presence.

Features:
-Pre-registered database of students.
-Automated attendance marking.
-Eliminates proxy attendance.
-Live facial recognition.-
-Secure QR code authentication

System Architecture:
-QR Code Generation Module: Creates dynamic QR codes for each session.
-Face Recognition Module: Captures and verifies students' faces.
-Attendance Validation Module: Matches the scanned QR code and face data.
-Student Records Database: Stores student details and images for face recognition.
-Attendance Logs: Maintains real-time attendance records.

Libraries Used:
-OpenCV – Face detection using Haar Cascade classifiers
-NumPy – Efficient matrix operations for image processing
-CSV – Storing and managing attendance records
-DateTime – Recording attendance timestamps
-Pyzbar – Scanning and decoding QR codes
-QR Code – Generating dynamic QR codes for attendance marking
