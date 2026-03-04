import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Optimized import logic for MediaPipe
@st.cache_resource
def load_segmentor():
    try:
        import mediapipe as mp
        # Accessing solutions directly
        mp_selfie = mp.solutions.selfie_segmentation
        return mp_selfie.SelfieSegmentation(model_selection=0)
    except Exception as e:
        st.error(f"Critical Error: Could not initialize MediaPipe. {e}")
        return None

segmentor = load_segmentor()

# Stop the app if segmentor is not initialized
if segmentor is None:
    st.info("Check your Streamlit Python version. 3.11 is highly recommended.")
    st.stop()

st.title("Code Sauce Labs: AI Background Remover")
st.write("Day 02 AI Challenge: Clean Background Extraction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    img_array = np.array(raw_image)
    
    # Ensure RGB format for MediaPipe
    if len(img_array.shape) > 2 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2: # Gray to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # AI processing to get the binary mask
    results = segmentor.process(img_array)
    
    # Logic: Masking the background based on confidence values
    # 
    mask = results.segmentation_mask > 0.1
    condition = np.stack((mask,) * 3, axis=-1)
    
    # Create black background
    bg_image = np.zeros(img_array.shape, dtype=np.uint8)
    
    # Merge using NumPy where logic
    output_image = np.where(condition, img_array, bg_image)
    
    # Display the result
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(output_image, caption="AI Result (Black BG)", use_container_width=True)
