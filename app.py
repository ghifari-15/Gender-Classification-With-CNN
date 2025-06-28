import cv2.data
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import tempfile
import threading


st.set_page_config(page_title="Gender Classification App", layout="centered")


# Load model
@st.cache_resource
def load_gender_model():
    return load_model("my_model.h5")


model = load_gender_model()

st.title("Gender Classification App")
menu = st.sidebar.selectbox("Menu", ["Upload Image", "Real Time Camera"])


# Helper for prediction
def predict_gender(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    if pred[0][0] >= 0.5:
        return "Female", float(pred[0][0]) * 100
    else:
        return "Male", (1 - float(pred[0][0])) * 100



if menu == "Upload Image":
    st.header("Upload Image for Gender Prediction")
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
       for (i, (uploaded_file)) in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            label, confidence = predict_gender(img_np)
            st.success(f"Prediction for {i+1}: {label} ({confidence:.2f}%)")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

        
      


# Real Time Camera menu with streamlit-webrtc for better webcam support
elif menu == "Real Time Camera":
    st.header("Real Time Gender Detection from Camera")
    try:
        from streamlit_webrtc import webrtc_streamer
        import av
    except ImportError:
        st.error(
            "streamlit-webrtc belum terinstall. Jalankan: pip install streamlit-webrtc av"
        )
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for x, y, w, h in faces:
                face_img = img[y : y + h, x : x + w]
                label, confidence = predict_gender(face_img)
                color = (0, 0, 255) if label == "Female" else (255, 0, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    img,
                    f"{label} ({confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                if label == "Male":
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(
                        img,
                        "MONITOR",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="gender", video_frame_callback=video_frame_callback)
