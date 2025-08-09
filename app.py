import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from collections import OrderedDict
import numpy as np
import math

# Initialize the Flask application
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Load the YOLOv8 model
# Make sure your custom-trained 'best.pt' is in the 'model/' folder
model = YOLO('model/best.pt')

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

class PotholeTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared # How many frames to wait before deregistering

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detected_centroids):
        if len(detected_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in detected_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances between tracked and detected centroids
            D = np.array([[euclidean_distance(obj_c, det_c) for det_c in detected_centroids] for obj_c in object_centroids])

            # Find the smallest distance for each tracked object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update the centroid
                object_id = object_ids[row]
                self.objects[object_id] = detected_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Handle disappeared objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            for col in unused_cols:
                self.register(detected_centroids[col])
                
        return self.objects

def process_video_and_count_potholes(video_path):
    """
    Processes the video to detect and track potholes, returning the total count of unique potholes.
    """
    cap = cv2.VideoCapture(video_path)
    tracker = PotholeTracker(max_disappeared=15) # Wait 15 frames before giving up on a pothole
    
    # This set will store the unique IDs of potholes that have been confirmed
    confirmed_pothole_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        results = model(frame)
        
        current_centroids = []
        for result in results:
            for box in result.boxes:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Calculate the center of the box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_centroids.append((center_x, center_y))

        # Update the tracker with the detected potholes
        tracked_objects = tracker.update(current_centroids)

        # Add the IDs of currently tracked objects to our confirmed set
        for object_id in tracked_objects.keys():
            confirmed_pothole_ids.add(object_id)

    cap.release()
    
    # The total count is the number of unique IDs we have ever confirmed
    return len(confirmed_pothole_ids)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            # Use the new, smarter counting function
            pothole_count = process_video_and_count_potholes(video_path)
            
            return render_template('index.html', pothole_count=pothole_count, video_path=filename)
            
    return render_template('index.html', pothole_count=None)

if __name__ == '__main__':
    app.run(debug=True)