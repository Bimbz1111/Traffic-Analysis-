import cv2
import os
import torchvision
import ultralytics
from collections import Counter
from ultralytics import YOLO
from collections import Counter
import torchaudio
import tempfile
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import Video

# Load the model
model = YOLO('yolov8n.pt')
model.save('model.pt')

# Start video capture
# cap = cv2.VideoCapture('moving_vehicle_vid_1.mp4')
# cap = cv2.VideoCapture(0)  # 0 for default webcam

# #while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Perform object detection
#     results = model(frame, verbose=False)[0]

#     # Visualize the results
#     cv2.imshow('Object Detection', results[0].plot())

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# video_path = "traffic_vid_3.mp4"
# Video(video_path, width=500, height=300)

#import #cv2

from collections import Counter
from ultralytics import YOLO

# Define classes of interest that we intend to track
classes = ["person", "bicycle", "car", "bus", "motorcycle", "motorbike"]



def process_frame(frame, model):
    results = model(frame, verbose=False)[0]

    # Extract detected object names and counts
    detected_classes = results.names  # This contains the object names
    class_indices = results.boxes.cls  # Indices for the detected classes
    detected_object_names = [detected_classes[int(index)] for index in class_indices] # Convert the class indices to object names
    object_counts = Counter(detected_object_names)

    # Determine traffic sign recommendation based on object counts
    vehicles = sum(count for name, count in object_counts.items() if name != "person")
    pedestrians = object_counts["person"]

    if vehicles > 10:
        message = "Traffic Sign: Green (Allow cars to pass)"
        msg_color = (0, 255, 0)
    elif vehicles > 0 and pedestrians > 5:
        message = "Traffic Sign: Yellow (Prepare to stop)"
        msg_color = (0, 255, 255)
    elif pedestrians > 0:
        message = "Traffic Sign: Red (Stop for pedestrians)"
        msg_color = (0, 0, 255)
    else:
        message = "Traffic Sign: Yellow (Prepare to stop)"
        msg_color = (0, 255, 255)

    # Draw recommendation text on the frame
    frame = cv2.rectangle(frame, (10, 10), (1500, 80), (0, 0, 0), -1)
    frame = cv2.putText(frame, f"Recommendation: {message}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, msg_color, 2)

    # Process detected objects (intruders)
    for i, box in enumerate(results.boxes.xyxy): # Use xyxy format for easier access
        class_id = int(results.boxes.cls[i])  # Directly use the class index
        class_name = results.names[class_id]  # Get the class name using the index
        confidence = results.boxes.conf[i].item()

# if class_id in classes:
        if class_name in classes:
            x_min, y_min, x_max, y_max = box.numpy().astype(int)
            text = f"{class_name} detected with {confidence*100:.2f}% certainty"
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            frame = cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (125, 246, 55), 1)

    return frame



# Use the existing model and classes variable from previous cells

# Capture video
# from IPython.display import Video
# video_path = "traffic_vid_4.mp4"
# Video(video_path, width=500, height=300)
# cap = cv2.VideoCapture("traffic_vid_4.mp4")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process the frame and display it
#     frame = cv2.resize(frame, (1900, 1000))
#     processed_frame = process_frame(frame)
#     cv2.imshow("Traffic Analysis System", processed_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# streamlit UI
st.title(" AI-powered Traffic Analysis system")
# Folder containing videos
video_folder = 'Videos/'  # Adjust this path as needed

# List all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

# Sidebar menu - each video as a selectable option
selected_video = st.sidebar.selectbox("Select a video to process:", video_files)

#st.write("upload a video to analyze traffic conditions and get real-time recordings")
#uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])



if selected_video:
    video_path = os.path.join(video_folder, selected_video)
    cap = cv2.VideoCapture(video_path)

# if uploaded_file is not None:
#     # Save uploaded file to temp file
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
    
    #cap = cv2.VideoCapture(tfile.name)
    
    # Prepare an output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "processed_output.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    stframe = st.empty()
    st.write(f"Processing {selected_video} ...")


    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Run detection
        processed_frame = process_frame(frame, model)
        # Write frame to output video
        out.write(processed_frame)
        # Display frame in Streamlit
        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    out.release()
    
    st.success(f"Finished processing {selected_video}.")
    #st.success("Processing complete. Download the output video:")
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, "processed_output.mp4")

    cap.release()
    cv2.destroyAllWindows()

    
    # def preprocess(image):
    #     # Convert the frame to tensor format, normalize, resize, etc.
    #     transform = T.Compose([
    #         T.ToPILImage(),
    #         T.Resize((640, 640)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     input_tensor = transform(image)
    #     return input_tensor

# def display_results(frame, output):
#     # Draw bounding boxes or other detection outputs on the frame
#     # Convert RGB back to BGR for OpenCV display
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     st.image(frame, caption='Processed Video', use_column_width=True)