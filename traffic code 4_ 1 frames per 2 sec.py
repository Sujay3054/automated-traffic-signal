import cv2
import torch
from ultralytics import YOLO

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# Video source
video_source = r"d:\codeathon 3.0\Ambulance_On_Empty_Road.mp4"
cap = cv2.VideoCapture(video_source)

# FPS handling (fallback if FPS can't be read)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frames_to_skip = 2  # Process every 2nd frame for real-time speed

# Traffic logic
signal_status = "RED"
frame_count = 0
VEHICLE_LIMIT = 6  # <-- change this threshold as you like

# Store the latest annotated frame so the video doesn't freeze or drop boxes
current_annotated_frame = None

def analyze_frame(frame):
    """
    Returns (vehicle_count, emergency_present, annotated_frame)
    """
    # imgsz=320 speeds up inference by ~3x!
    # verbose=False removes the print spam in terminal
    results = model.predict(frame, imgsz=320, conf=0.3, verbose=False)

    vehicle_count = 0
    emergency_present = False

    # Common COCO vehicle classes
    vehicle_labels = {"car", "truck", "bus", "motorbike", "bicycle"}
    emergency_labels = {"ambulance", "police car", "police", "fire truck"}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))

            if label in vehicle_labels:
                vehicle_count += 1
            if label in emergency_labels:
                emergency_present = True

    # plot() draws the bounding boxes on the frame!
    annotated_frame = results[0].plot()

    return vehicle_count, emergency_present, annotated_frame


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame every `frames_to_skip` frames
        if frame_count % frames_to_skip == 0:
            vehicle_count, emergency_present, current_annotated_frame = analyze_frame(frame)

            if emergency_present:
                signal_status = "GREEN"
            else:
                signal_status = "GREEN" if vehicle_count >= VEHICLE_LIMIT else "RED"

        # Use the latest annotated frame, or the raw frame if none exists yet
        display_frame = current_annotated_frame.copy() if current_annotated_frame is not None else frame.copy()

        # Display signal status and vehicle count
        color = (0, 255, 0) if signal_status == "GREEN" else (0, 0, 255)
        display_text = f"Signal: {signal_status}"
        if emergency_present:
            display_text += " (Ambulance Detected)"

        cv2.putText(
            display_frame,
            display_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.putText(
            display_frame,
            f"Vehicles: {vehicle_count if 'vehicle_count' in locals() else 0}/{VEHICLE_LIMIT}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Traffic - NORTH", display_frame)
        frame_count += 1

        # Playback speed (1ms wait allows it to play as fast as possible for real-time feel)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
