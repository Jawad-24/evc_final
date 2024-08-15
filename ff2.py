import firebase_admin
from firebase_admin import credentials, storage, db
import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/Users/alaaalseeni/Desktop/final/final-ce293-firebase-adminsdk-2esv3-2afa5fc7b3.json')  # Replace with your service account key file
firebase_admin.initialize_app(cred, {
    'storageBucket': 'final-ce293.appspot.com',  # Replace with your Firebase Storage bucket name
    'databaseURL': 'https://final-ce293-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})

# Initialize YOLO model
model = YOLO('/Users/alaaalseeni/Desktop/final/best (5).pt')  # Replace with the path to your YOLO model

# Define local directory to save videos
local_directory = '/Users/alaaalseeni/Desktop/fina'
os.makedirs(local_directory, exist_ok=True)

# Get a reference to the storage bucket
bucket = storage.bucket()

# List and download all videos from Firebase Storage
blobs = bucket.list_blobs()

for blob in blobs:
    if blob.name.endswith('.mov'):  # Process only .mp4 files
        local_file_path = os.path.join(local_directory, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)
        print(f'Downloaded: {local_file_path}')
        
        # Process the downloaded video
        cap = cv2.VideoCapture(local_file_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        red_line = int(frame_height * 1/3)
        blue_line = int(frame_height * 2/3)
        count = 0
        passed_red_line = False
        passed_blue_line = False
        is_in = True

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Perform YOLOv8 inference on the frame
            results = model(frame)

            cv2.line(frame, (0, red_line), (frame_width, red_line), (0, 0, 255), 3)  # red line
            cv2.line(frame, (0, blue_line), (frame_width, blue_line), (255, 0, 0), 3)  # blue line

            for result in results:
                for box in result.boxes:
                    conf = box.conf[0].item()  # Confidence score
                    if conf > 0.85:  # Confidence threshold
                        class_id = int(box.cls[0].item())  # Class ID
                        label = f'Class {class_id}'  # Replace with your actual class labels
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of detected object
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                        cv2.putText(frame, f'{label} detected {conf:.2f}', (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                        if (red_line - 20) < cy < (red_line + 20):
                            passed_red_line = True
                            if not passed_blue_line:
                                is_in = False

                        if (blue_line - 20) < cy < (blue_line + 20):
                            passed_blue_line = True

                        if passed_blue_line and passed_red_line:
                            count = 1 if is_in else -1
                            new_row = {
                                "product_id": class_id,
                                "is_add": is_in,
                                "count": count,
                                "date": datetime.now().isoformat()
                            }
                            print(new_row)

                            try:
                                db.reference('object_counts').push(new_row)
                                print("Data successfully pushed to Firebase")
                            except Exception as e:
                                print("Failed to push data to Firebase:", e)

                            passed_red_line, passed_blue_line = False, False
                            is_in = True

        cap.release()
        cv2.destroyAllWindows()