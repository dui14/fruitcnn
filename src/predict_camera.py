import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os
import time

# CẤU HÌNH
IMG_SIZE = (100, 100)
MODEL_PATH = '../models/model/fruit_model_full.h5'  # Đường dẫn model
FRUIT_INFO_PATH = '../data/fruit_info.json'  # Đường dẫn json
FRUIT_CLASSES_PATH = '../data/fruit_classes.txt'  # Đường dẫn class names

# LOAD DỮ LIỆU PHỤ
if os.path.exists(FRUIT_CLASSES_PATH):
    with open(FRUIT_CLASSES_PATH, 'r') as f:
        FRUIT_CLASSES = [line.strip() for line in f.readlines()]
else:
    FRUIT_CLASSES = ["Apple", "Banana", "Orange"]

if os.path.exists(FRUIT_INFO_PATH):
    with open(FRUIT_INFO_PATH, 'r') as f:
        FRUIT_INFO = json.load(f)
else:
    FRUIT_INFO = {
        "Apple": {"calories": 52, "desc": "giàu Vitamin C và chất xơ"},
        "Banana": {"calories": 89, "desc": "giàu Kali"},
        "Orange": {"calories": 47, "desc": "siêu giàu Vitamin C"}
    }

# LOAD MODEL
try:
    model = load_model(MODEL_PATH)
    print(f" Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f" Không load được model → Dùng DUMMY MODE ({e})")
    model = None

QUALITY_LABELS = ['A', 'B', 'C']
DEFECT_LABELS = ['Không có', 'Vết dập nhẹ', 'Có 2 vết đen', 'Mốc']

# HÀM TIỀN XỬ LÝ
def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# DỰ ĐOÁN 
CONFIDENCE_THRESHOLD = 0.70

def predict(frame):
    if model is None:
        # Dummy mode nếu sai model
        import random
        fruit_idx = random.randint(0, len(FRUIT_CLASSES)-1)
        fruit_prob = random.uniform(0.85, 0.99)
        quality_idx = random.randint(0, 2)
        defect_idx = 0
        max_confidence = fruit_prob
    else:
        img = preprocess_frame(frame)
        try:
            outputs = model.predict(img, verbose=0)
            fruit_probs = outputs[0]
            # quality_probs = outputs[1][0]
            # defect_probs = outputs[2][0]

            fruit_idx = np.argmax(fruit_probs)
            fruit_prob = fruit_probs[fruit_idx]
            quality_idx = 0
            defect_idx = 0
            max_confidence = fruit_prob
        except Exception as e:
            print(f"Lỗi predict: {e}")
            return None

    if max_confidence < CONFIDENCE_THRESHOLD:
        return {"error": "low_confidence"}

    raw_class = FRUIT_CLASSES[fruit_idx]
    fruit_name = raw_class.split(' - ')[0] if ' - ' in raw_class else raw_class
    nutri = FRUIT_INFO.get(fruit_name, {"calories": 0, "desc": "không rõ"})

    return {
        "fruit": f"{fruit_name} ({fruit_prob:.0%})",
        "nutrition": f"{nutri['calories']} kcal/100g, {nutri['desc']}",
        "quality": f"Loại {QUALITY_LABELS[quality_idx]}",
        "defect": DEFECT_LABELS[defect_idx]
    }

# VẼ KẾT QUẢ
def draw_result(frame, result, x=10, y=30):
    if result is None:
        return
    lines = [
        result["fruit"],
        result["nutrition"],
        result["quality"],
        result["defect"]
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x, y + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# MAIN LOOP
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera đã mở. Hãy đưa mẫu vật vào khung để quét.")
    print("Nhấn 'q' để thoát.")

    last_predict_time = 0
    predict_interval = 0.5
    current_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        x1, y1 = margin_x, margin_y
        x2, y2 = w - margin_x, h - margin_y

        # Khung mặc định ĐỎ
        border_color = (0, 0, 255)
        warning_text = "Hay dua mau vat vao Camera"
        show_warning = True

        # Lấy vùng quét
        scan_zone = frame[y1:y2, x1:x2]
        if scan_zone.size == 0:
            scan_zone = frame

        current_time = time.time()
        if current_time - last_predict_time > predict_interval:
            result = predict(scan_zone)
            last_predict_time = current_time

            if result is not None and "error" not in result:
                current_result = result
                border_color = (0, 255, 0)  # Xanh khi có trái cây
                show_warning = False
            else:
                current_result = None

        # Vẽ khung quét
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 3)

        # Hiển thị dòng hướng dẫn
        if show_warning:
            cv2.putText(frame, warning_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Hiển thị kết quả nếu có
        draw_result(frame, current_result)

        # Hiển thị FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Fruit Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
