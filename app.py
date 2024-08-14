from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import math
# Initialize Firebase Admin SDK
cred = credentials.Certificate('final-ce293-firebase-adminsdk-2esv3-30c9ea5b8e.json')  # Replace with the path to your service account key file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://final-ce293-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs('./static/uploads', exist_ok=True)

# Load the YOLO model
model = YOLO('besst.pt')
class_labels = {
    0: 'جبنة كيري', 1: 'سالمون مدخن', 2: 'بيض', 3: 'حليب', 4: 'زبدة', 
    5: 'سجق', 6: 'افكادو', 7: 'طماطم', 8: 'لبن', 9: 'كيوي', 
    10: 'لبنة', 11: 'دجاج', 12: 'رمان'
}

@app.route('/restaurantpage')
def index():
    coef1 = -3.00092659
    coef2 =  1.00115665
    intercept = 2.015704020036677518
    lst = []
    for k,v in class_labels.items():
        pred = (int(coef1*k + coef2*9 + intercept))
        if pred < 0:
            pred = -pred
        lst.append({"اسم المنتج":class_labels[k], "الكمية المقترحة":pred})
    # print(lst)
    df = pd.DataFrame(lst)
    df_html = df.to_html(classes='table table-striped', index=False)
    return render_template('forecasting.html',  table=df_html)

@app.route('/')
def restaurantpage():    
    ref = db.reference('/dht11')
    data = ref.get()
    #print(list(data.values())[-1])
    temperature = list(data.values())[-1]['temperature']
    humidity = list(data.values())[-1]['humidity']
    tom = (temperature)
    hum = (humidity)
    data = [temperature, humidity]
    return render_template('restaurantpage.html', data=data)


@app.route('/inventory')
def inventory():


    try:
        ref = db.reference('/object_counts')
        data = ref.get()

        if data:
            df = pd.DataFrame(list(data.values()))
            df = df.groupby('product_id')[['count']].sum().reset_index()
            df['product_id'] = df['product_id'].apply(lambda x: class_labels[x])
            df = df.rename(columns={"product_id": "اسم المنتج", "count":"الكمية"})
            df_html = df.to_html(classes='table table-striped', index=False)
        else:
            df_html = "<p>No data available.</p>"
    except Exception as e:


        df_html = f"<p>Error fetching data: {str(e)}</p>"
    return render_template('Inventory.html', table=df_html)

@app.route('/get_inventory_data', methods=['GET'])
def get_inventory_data():
    ref = db.reference('object_counts')
    data = ref.get()
    if data is None:
        data = {}
    return jsonify(data)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    red_line = int(frame_height * 1 / 3)
    blue_line = int(frame_height * 2 / 3)
    text_color = (255, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    count = 0
    passed_red_line = False
    passed_blue_line = False
    is_in = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame)

        cv2.line(frame, (0, red_line), (frame_width, red_line), red_color, 3)
        cv2.line(frame, (0, blue_line), (frame_width, blue_line), blue_color, 3)

        for info in results:
            if len(info.boxes.conf) != 0 and info.boxes.conf[0] > 0.5:  # Adjusted threshold
                class_id = int(info.boxes.cls[0])
                label = class_labels[class_id]
                conf = info.boxes.conf[0]
                x1, y1 = int(info.boxes.xyxy[0][0]), int(info.boxes.xyxy[0][1])
                x2, y2 = int(info.boxes.xyxy[0][2]), int(info.boxes.xyxy[0][3])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'{label} detected {conf:.2f}', (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                if red_line - 20 < cy < red_line + 20:
                    passed_red_line = True
                    if not passed_blue_line:
                        is_in = False
                if blue_line - 20 < cy < blue_line + 20:
                    passed_blue_line = True

                if passed_blue_line and passed_red_line:
                    count = 1 if is_in else -1
                    new_row = {
                        'product_id': class_id,
                        'is_add': is_in,
                        'count': count,
                        'date': datetime.now().isoformat()
                    }
                    db.reference('object_counts').push(new_row)
                    passed_red_line, passed_blue_line = False, False
                    is_in = True

        cv2.putText(frame, f'COUNT: {count}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
