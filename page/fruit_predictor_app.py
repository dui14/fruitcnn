import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Import TensorFlow với try-except để dễ debug nếu lỗi môi trường
try:
    import tensorflow as tf
except Exception as e:
    st.error(f"Lỗi import TensorFlow: {str(e)}. Hãy cài lại numpy==1.26.4 và tensorflow==2.15.0")
    raise

# Load models and classes
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('fruit_model_full.h5')

@st.cache_resource
def load_yolo_model():
    return YOLO('./best.pt')

@st.cache_data
def load_classes():
    with open('fruit_classes.txt', 'r') as f:
        return [line.strip().split(': ')[1] for line in f.readlines()]

# CNN prediction
def predict_cnn(image, model, classes):
    img = image.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return classes[predicted_class], confidence

# YOLO prediction (single image)
def predict_yolo(image, model):
    img_array = np.array(image)
    results = model(img_array, conf=0.5, verbose=False)
    if len(results[0].boxes) > 0:
        det = results[0].boxes[0]
        cls = int(det.cls[0].cpu().numpy())
        conf = det.conf[0].cpu().numpy()
        class_name = model.names[cls]
        return class_name, conf
    return "Unknown", 0.0

# Ensemble
def ensemble_predict(image, cnn_model, yolo_model, classes):
    cnn_class, cnn_conf = predict_cnn(image, cnn_model, classes)
    yolo_class, yolo_conf = predict_yolo(image, yolo_model)
    
    if cnn_class.lower() == yolo_class.lower():
        combined_conf = (cnn_conf + yolo_conf) / 2
        return cnn_class, combined_conf
    else:
        return cnn_class if cnn_conf > yolo_conf else yolo_class, max(cnn_conf, yolo_conf)

# Real-time video processor (YOLO)
class VideoProcessor:
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.yolo_model(img, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        frame_h, frame_w = img.shape[:2]
        box_w, box_h = 250, 250
        x = (frame_w - box_w) // 2
        y = (frame_h - box_h) // 2
        cv2.rectangle(annotated_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ====================== STREAMLIT APP ======================
st.title("🍎 Nhận Diện Trái Cây - CNN + YOLO Ensemble")

option = st.sidebar.selectbox("Chọn chế độ", ["Upload Hình Ảnh", "Camera Real-Time"])

cnn_model = load_cnn_model()
yolo_model = load_yolo_model()
classes = load_classes()

if option == "Upload Hình Ảnh":
    uploaded_file = st.file_uploader("Chọn hình ảnh...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Hình ảnh đã upload', use_column_width=True)
        
        with st.spinner('Đang nhận diện bằng cả CNN và YOLO...'):
            fruit_name, confidence = ensemble_predict(image, cnn_model, yolo_model, classes)
            
            st.success(f"**Trái cây:** {fruit_name}")
            st.info(f"**Độ tin cậy (ensemble):** {confidence:.2%}")
            
            if confidence > 0.8:
                st.balloons()

elif option == "Camera Real-Time":
    st.write("**Real-time detection bằng YOLO** (đặt trái cây vào khung giữa)")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="yolo_camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=lambda: VideoProcessor(yolo_model),
        async_processing=True,
    )

                