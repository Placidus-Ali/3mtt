import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import Sort
import torch
print("Libraries Imported")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize SORT tracker
tracker = Sort()

# Streamlit app
st.title("Traffic Monitoring: Object Tracking and Vehicle Counting")
st.sidebar.title("Upload Video")
print("Upload Video Created")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
counting_line = st.sidebar.slider("Set counting line (Y-axis)", min_value=50, max_value=700, value=300)

if uploaded_file is not None:
    # Temporary file for the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Vehicle count
    vehicle_count = 0
    tracked_ids = set()

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 360))

        # Detect objects
        results = model(resized_frame)
        detections = results.pandas().xyxy[0]  # Convert to Pandas DataFrame

        if not detections.empty:
            detection_array = detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values

            # Track objects
            tracked_objects = tracker.update(detection_array)

            # Draw tracking results
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = map(int, obj)
                label = f"ID {obj_id}"
                center_y = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Check if the object crosses the counting line
                if center_y > counting_line - 10 and center_y < counting_line + 10:
                    if obj_id not in tracked_ids:
                        vehicle_count += 1
                        tracked_ids.add(obj_id)

            # Draw counting line
            cv2.line(resized_frame, (0, counting_line), (640, counting_line), (0, 255, 255), 2)

            # Display vehicle count
            cv2.putText(resized_frame, f"Vehicle Count: {vehicle_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display frame in Streamlit
        stframe.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()
    st.success(f"Processing Complete! Total Vehicles Counted: {vehicle_count}")

else:
    st.info("Upload a video file to get started.")
print("Model working perfectly")