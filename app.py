import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Use direct import to bypass the AttributeError in specific environments
try:
    from mediapipe.python.solutions import selfie_segmentation as mp_selfie
except ImportError:
    st.error("MediaPipe internal modules not found. Please check your requirements.txt.")

# Initialize the segmentation model directly
segmentor = mp_selfie.SelfieSegmentation(model_selection=0)

st.title("AI Background Remover")
st.write("Upload an image to remove the background using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an Image object
    raw_image = Image.open(uploaded_file)
    img_array = np.array(raw_image)
    
    # MediaPipe works with RGB. If image is RGBA, convert it.
    if len(img_array.shape) > 2 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Process the image to get the segmentation mask
    results = segmentor.process(img_array)
    
    # Generate binary mask (person vs background)
    # Threshold set at 0.1 for better edge detection
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    
    # Create a black background of the same size
    bg_image = np.zeros(img_array.shape, dtype=np.uint8)
    
    # Apply the mask: Keep original where mask is True, otherwise use black
    output_image = np.where(condition, img_array, bg_image)
    
    # Display the results side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(output_image, caption="AI Result (Black BG)", use_container_width=True)
