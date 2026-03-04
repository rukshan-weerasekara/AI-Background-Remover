import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Use the standard import first
try:
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation
    segmentor = mp_selfie.SelfieSegmentation(model_selection=0)
except Exception as e:
    st.error(f"MediaPipe Loading Error: {e}")
    st.info("Please ensure your Python version is set to 3.11 in Streamlit Settings.")
    st.stop() # Stops execution to prevent NameError

st.title("Code Sauce Labs: AI Background Remover")
st.write("Upload an image to remove the background using AI.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    img_array = np.array(raw_image)
    
    # Handle RGBA images
    if len(img_array.shape) > 2 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # AI Processing
    results = segmentor.process(img_array)
    
    # Masking logic
    # 
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(img_array.shape, dtype=np.uint8)
    output_image = np.where(condition, img_array, bg_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(output_image, caption="AI Result (Black BG)", use_container_width=True)
