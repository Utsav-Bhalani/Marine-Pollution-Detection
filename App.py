import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import tempfile
import cv2

# Define custom CSS for a complete UI overhaul with beautiful, colorful background
st.markdown("""
    <style>
        /* Global Background */
        .main {
            background: linear-gradient(135deg, #ff6a00, #ee0979);
            color: white;
            font-family: 'Arial', sans-serif;
            padding: 0;
            margin: 0;
        }

        /* Sidebar Background */
        .sidebar .sidebar-content {
            background-color: #0a0a0a;
            color: #fff;
        }

        .sidebar .sidebar-header {
            background-color: #333;
            color: #fff;
            font-weight: bold;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #ff4c4c;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 12px;
            padding: 12px 24px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #f44336;
        }

        /* File Uploader */
        .stFileUploader>div>button {
            background-color: #1f90a3;
            color: white;
            border-radius: 8px;
        }
        .stFileUploader {
            background-color: #0e76a8;
        }

        /* Selectbox & Radio Buttons Background */
        .stSelectbox, .stRadio {
            font-size: 14px;
            background-color: #333;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }

        /* Selectbox Styling */
        .stSelectbox>div {
            background-color: #444;
            padding: 10px;
            border-radius: 10px;
        }

        .stSelectbox>div>div {
            background-color: #444;
            color: #ffffff;
        }

        .stSelectbox select {
            background-color: #444;
            border: 1px solid #ff6a00;
            color: #fff;
            font-size: 14px;
            padding: 6px 12px;
            border-radius: 10px;
        }

        .stSelectbox select:hover {
            background-color: #555;
        }

        /* Radio Buttons Styling */
        .stRadio>div>div>span {
            color: #ffffff;
            background-color: #444;
        }

        .stRadio>div>div:hover {
            background-color: #555;
        }

        .stRadio input[type="radio"] {
            background-color: #444;
            border: 1px solid #ff6a00;
            color: #fff;
            font-size: 14px;
            padding: 6px 12px;
            border-radius: 10px;
        }

        /* Image Styling */
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Heading Styling */
        h1 {
            color: #ff6a00;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }

        h2 {
            color: #ee0979;
            font-size: 24px;
            text-align: center;
        }

        .stMarkdown>div {
            color: #fff;
        }

        /* Output Container */
        .st-container {
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Define the paths for your models
model_paths = {
    "YOLO V8 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8_model.pt',
    "YOLO V8 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8small_model.pt',
    "YOLO V8 MEDIUM": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8m_model.pt',
    "YOLO V10 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10nano_model-2 (1).pt',
    "YOLO V10 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10small_model.pt',
}

# Streamlit title and introductory text with styled headers
st.title("🌊 Marine Pollution Detection")
st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #eee;'> 
        Detect and classify marine pollution using advanced YOLO models.
    </div>
""", unsafe_allow_html=True)

# Sidebar for model selection and input type
st.sidebar.title("Choose Model & Input")
model_choice = st.sidebar.selectbox("Select YOLOv8 Model:", list(model_paths.keys()),
                                    help="Choose a YOLOv8 model for detection.")
input_type = st.sidebar.radio("Select Input Type:", ('Image', 'Video'), help="Choose the type of input for detection.")

# Load the selected model
model = YOLO(model_paths[model_choice])

# Container for input section
input_container = st.container()

# Image detection
if input_type == 'Image':
    uploaded_file = input_container.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"],
                                                  label_visibility="collapsed")

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        input_container.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform detection
        results = model(image)

        # Access the first result from the list
        result = results[0]

        # Create an 'output' folder if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Annotate the image
        annotated_image = result.plot()
        annotated_pil_image = Image.fromarray(np.uint8(annotated_image))

        # Save the annotated image
        save_path = os.path.join(output_dir, 'annotated_image.jpg')
        annotated_pil_image.save(save_path)

        # Output Section: Displaying Results Below
        output_container = st.container()

        with output_container:
            st.subheader("🔍 Detection Results")
            st.write(f"Detected Objects (Bounding Boxes): {result.boxes.xyxy}")
            st.write(f"Confidence Scores: {result.boxes.conf}")
            st.write(f"Class Names: {result.names}")

            # Display the annotated image
            st.image(save_path, caption='Detected Image with Bounding Boxes', use_column_width=True)

# Video detection
elif input_type == 'Video':
    uploaded_video = input_container.file_uploader("Upload a Video...", type=["mp4", "avi", "mov"],
                                                   label_visibility="collapsed")

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Open the video using OpenCV
        video_capture = cv2.VideoCapture(video_path)

        # Create an 'output' folder if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Define video writer to save the annotated video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        out_path = os.path.join(output_dir, 'annotated_video.mp4')
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

        stframe = st.empty()  # Placeholder for displaying video frames

        # Output Section for video processing
        output_container = st.container()

        # Process video frame by frame
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame to RGB (YOLO expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform detection
            results = model(frame_rgb)

            # Annotate the frame
            annotated_frame = results[0].plot()

            # Write annotated frame to output video
            out.write(annotated_frame)

            # Display the annotated frame
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        # Release resources
        video_capture.release()
        out.release()

        # Output path of the video
        st.write(f"Annotated video saved at: {out_path}")

# import necessary modules
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np
# import os
# import tempfile
# import cv2
#
# # Import Grad-CAM module
# from gradcam import generate_gradcam  # Ensure `generate_gradcam` is defined in `gradcam.py`
#
# # Define paths for your models
# model_paths = {
#     "YOLO V8 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8_model.pt',
#     "YOLO V8 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8small_model.pt',
#     "YOLO V8 MEDIUM": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8m_model.pt',
#     "YOLO V10 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10nano_model-2 (1).pt',
#     "YOLO V10 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10small_model.pt',
# }
#
# # Streamlit title
# st.title('Marine Pollution Detection with Grad-CAM Visualization')
#
# # Dropdown to select model
# model_choice = st.selectbox("Choose a YOLOv8 model:", list(model_paths.keys()))
#
# # Load the selected model
# model = YOLO(model_paths[model_choice])
#
# # Option to choose between image or video input
# input_type = st.radio("Choose input type", ('Image', 'Video'))
#
# # Image detection with Grad-CAM
# if input_type == 'Image':
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Load the image
#         image = Image.open(uploaded_file)
#
#         # Display the uploaded image
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#         st.write("Classifying...")
#
#         # Perform object detection
#         results = model(image)
#         result = results[0]
#
#         # Display detection results (bounding boxes, labels, confidence scores)
#         st.write("Bounding boxes:", result.boxes.xyxy)  # Coordinates of bounding boxes
#         st.write("Confidence scores:", result.boxes.conf)  # Confidence scores
#         st.write("Class names:", result.names)  # Class names
#
#         # Plot and save the detected results with bounding boxes
#         annotated_image = result.plot()
#         annotated_pil_image = Image.fromarray(np.uint8(annotated_image))
#
#         # Create output directory if it doesn't exist
#         output_dir = 'output'
#         os.makedirs(output_dir, exist_ok=True)
#         annotated_image_path = os.path.join(output_dir, 'annotated_image.jpg')
#         annotated_pil_image.save(annotated_image_path)
#
#         # Display the annotated image with bounding boxes
#         st.image(annotated_image_path, caption='Detected Image with Bounding Boxes', use_column_width=True)
#
#         # Generate and display Grad-CAM heatmap
#         gradcam_heatmap = generate_gradcam(model, image)
#         gradcam_heatmap_pil = Image.fromarray(np.uint8(gradcam_heatmap))
#         st.image(gradcam_heatmap_pil, caption="Grad-CAM Heatmap", use_column_width=True)
#
# # Video detection with Grad-CAM
# elif input_type == 'Video':
#     uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
#
#     if uploaded_video is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         video_path = tfile.name
#
#         video_capture = cv2.VideoCapture(video_path)
#
#         output_dir = 'output'
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Define video writer for annotated video output
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out_path = os.path.join(output_dir, 'annotated_video.mp4')
#         fps = int(video_capture.get(cv2.CAP_PROP_FPS))
#         frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
#
#         stframe = st.empty()  # Streamlit placeholder for video frame display
#
#         # Process each frame
#         while video_capture.isOpened():
#             ret, frame = video_capture.read()
#             if not ret:
#                 break
#
#             # Convert frame to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Perform detection on frame
#             results = model(frame_rgb)
#             annotated_frame = results[0].plot()
#
#             # Convert annotated frame for display and video output
#             annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
#             out.write(annotated_frame_bgr)
#
#             # Display the annotated frame in the Streamlit app
#             stframe.image(annotated_frame)
#
#         video_capture.release()
#         out.release()
#
#         # Display download link for the annotated video
#         st.video(out_path, format="video/mp4")
#         st.write(f"Annotated video saved at: {out_path}")
