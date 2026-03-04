# 🖼️ AI Background Remover (MediaPipe Segmentation)

An intelligent Computer Vision tool designed to instantly isolate subjects and remove backgrounds. This application leverages Google's MediaPipe technology to provide high-speed, professional-grade subject extraction.

## 🔗 Live Demo
Check out the live app here: [https://ai-background-remover-fxtnyrynn2xzga6ervjc33.streamlit.app/](https://ai-background-remover-fxtnyrynn2xzga6ervjc33.streamlit.app/)

## 🚀 Project Overview
As a **3D Animator and AI Developer**, I understand the importance of clean subject isolation in visual storytelling. I built this tool under **Code Sauce Labs** to streamline the process of background removal using **Selfie Segmentation**—a specialized branch of Computer Vision. 

### Key Features:
* **High-Speed Subject Isolation:** Uses the **MediaPipe Selfie Segmentation** model, optimized for real-time performance.
* **Precise Edge Detection:** Generates a binary mask to distinguish between the subject and the background with high confidence.
* **Clean Composition:** Automatically replaces the background with a solid aesthetic black layer, ready for further VFX or design work.
* **Format Flexibility:** Seamlessly processes standard image formats including **JPG, JPEG, and PNG**.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **AI Framework:** MediaPipe (Selfie Segmentation)
* **Image Processing:** OpenCV (cv2), NumPy, Pillow (PIL)
* **Frontend:** Streamlit
* **Deployment:** Streamlit Cloud / GitHub

## 🧠 The Logic
The system executes a precise Image Segmentation pipeline:
1.  **Preprocessing:** Converting the input image to the **RGB color space** to meet the AI model's requirements.
2.  **Mask Generation:** The AI analyzes the frame and produces a **Probability Mask** where each pixel is assigned a value based on the likelihood of it being part of the subject.
3.  **Thresholding & Merging:** Applying a threshold (e.g., $mask > 0.1$) to create a binary map, then using **NumPy** to merge the original pixels with a generated background.

---
Developed by **Rukshan Weerasekara** *Creative Technologist | 3D Animator | AI Developer* Founder of **Code Sauce Labs**
