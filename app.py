import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=0)

st.title("AI Background Remover")
st.write("Upload an image to remove the background using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # MediaPipe needs RGB, but if the image is RGBA, we convert it
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Process the image
    results = segmentor.process(img_array)
    
    # Create the mask
    mask = results.segmentation_mask > 0.1
    
    # Create a black background
    bg_image = np.zeros(img_array.shape, dtype=np.uint8)
    
    # Combine original image and background
    output_image = np.where(mask[:, :, None], img_array, bg_image)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(output_image, caption="AI Processed (Black BG)", use_column_width=True)
