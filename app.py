import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
import math

# Initialize the Flask application
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Load the YOLOv8 model
model = YOLO('model/best.pt')

# Function to calculate the Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            pothole_count = process_video(video_path)
            
            return render_template('index.html', pothole_count=pothole_count, video_path=filename)
            
    return render_template('index.html', pothole_count=None)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    tracked_potholes = {}
    pothole_id_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform pothole detection
        results = model(frame)
        
        current_potholes_centers = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_potholes_centers.append((center_x, center_y))

        # Simple tracking logic
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
                    if dist < 250 and dist < min_dist: # 50 pixels threshold for matching
                        min_dist = dist
                        best_match_center = current_center
                        found_match = True
                
                if found_match:
                    tracked_potholes[pothole_id] = best_match_center
                    newly_detected_centers.remove(best_match_center)

            # Add new potholes
            for new_center in newly_detected_centers:
                tracked_potholes[pothole_id_counter] = new_center
                pothole_id_counter += 1

    cap.release()
    return len(tracked_potholes)

if __name__ == '__main__':
    app.run(debug=True)