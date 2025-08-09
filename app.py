import os
import math
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 17 * 1024 * 1024  # 16MB max upload size

# Load YOLO model
model = YOLO('model/best.pt')

# Euclidean distance function
def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Process video
        pothole_count = process_video(video_path)

        return jsonify({
            'pothole_count': pothole_count,
            'video_path': filename
        })

    return render_template('index.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    tracked_potholes = {}
    pothole_id_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_potholes_centers = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_potholes_centers.append((center_x, center_y))

        if not tracked_potholes:
            for center in current_potholes_centers:
                tracked_potholes[pothole_id_counter] = center
                pothole_id_counter += 1
        else:
            newly_detected_centers = current_potholes_centers.copy()
            for pothole_id, last_center in list(tracked_potholes.items()):
                found_match = False
                min_dist = float('inf')
                best_match_center = None

                for current_center in newly_detected_centers:
                    dist = euclidean_distance(last_center, current_center)
                    if dist < 250 and dist < min_dist:
                        min_dist = dist
                        best_match_center = current_center
                        found_match = True

                if found_match:
                    tracked_potholes[pothole_id] = best_match_center
                    newly_detected_centers.remove(best_match_center)

            for new_center in newly_detected_centers:
                tracked_potholes[pothole_id_counter] = new_center
                pothole_id_counter += 1

    cap.release()
    return len(tracked_potholes)

if __name__ == '__main__':
    app.run(debug=True)
