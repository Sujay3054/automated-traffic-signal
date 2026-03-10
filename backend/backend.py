from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import torch
from ultralytics import YOLO
import threading
import time
import os

app = Flask(__name__)
CORS(app)

# Global variables to share data between threads
current_status = {
    'signal_status': 'RED',
    'vehicle_count': 0,
    'detected_labels': [],
    'frame': None,
    'is_running': False
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    frames_to_skip = fps * 2  
    frame_count = 0  

    def detect_vehicles(frame):
        results = model(frame)
        return sum(1 for result in results for box in result.boxes if model.names[int(box.cls[0])] in ["car", "bus", "truck", "motorcycle"])

    def detect_ambulance(frame):
        results = model(frame)
        detected_labels = []  
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                detected_labels.append(label)  
                if label in ["ambulance", "truck", "bus"]:
                    return True, detected_labels  
        return False, detected_labels 

    try:
        while current_status['is_running'] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            if frame_count % frames_to_skip == 0:
                vehicle_count = detect_vehicles(frame)
                ambulance_detected, detected_labels = detect_ambulance(frame)
                if ambulance_detected:
                    current_status['signal_status'] = 'GREEN'
                elif vehicle_count >= 8:
                    current_status['signal_status'] = 'GREEN'
                else:
                    current_status['signal_status'] = 'RED'
                
                current_status['vehicle_count'] = vehicle_count
                current_status['detected_labels'] = detected_labels

            # Encode frame for web display
            _, buffer = cv2.imencode('.jpg', frame)
            current_status['frame'] = buffer.tobytes()
            frame_count += 1

            time.sleep(0.03)  # Control processing speed

    finally:
        cap.release()
        current_status['is_running'] = False

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'signal_status': current_status['signal_status'],
        'vehicle_count': current_status['vehicle_count'],
        'detected_labels': current_status['detected_labels']
    })

@app.route('/api/video_feed', methods=['GET'])
def video_feed():
    if current_status['frame'] is None:
        return "No video available", 404
    return current_status['frame'], 200, {'Content-Type': 'image/jpeg'}

@app.route('/api/start', methods=['POST'])
def start_processing():
    if current_status['is_running']:
        return jsonify({'message': 'Already running'}), 200
    
    video_source = request.json.get('video_source', 0)  # Default to webcam
    if isinstance(video_source, str) and not os.path.exists(video_source):
        return jsonify({'error': 'Video file not found'}), 404
    
    current_status['is_running'] = True
    thread = threading.Thread(target=process_video, args=(video_source,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Processing started'}), 200

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    current_status['is_running'] = False
    return jsonify({'message': 'Processing stopped'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

