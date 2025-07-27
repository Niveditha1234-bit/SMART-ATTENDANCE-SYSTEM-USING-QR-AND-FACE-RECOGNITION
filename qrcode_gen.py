import qrcode
import os
from datetime import datetime

def generate_qr():
    room_no = input("Enter Room Number: ")
    period_no = input("Enter Period Number: ")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    qr_data = f"Room: {room_no}, Period: {period_no}, Timestamp: {timestamp}"
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill="black", back_color="white")
    
    # Create the qrcodes folder if it doesn't exist
    folder_path = "qrcodes"
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"QR_Room{room_no}_Period{period_no}.png")
    img.save(file_path)
    
    print(f"QR Code generated and saved at {file_path}")

# Run the function
generate_qr()
