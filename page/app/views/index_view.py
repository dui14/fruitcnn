from flask import Blueprint, render_template, request, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = 'best.pt'
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

index_bp = Blueprint('index', __name__)

camera = None
camera_active = False

def get_camera():
    """Khởi tạo camera nếu chưa có"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    """Giải phóng camera"""
    global camera, camera_active
    if camera is not None:
        camera.release()
        camera = None
    camera_active = False

def draw_detection_frame(frame, x, y, w, h):
    """Vẽ khung nhận diện"""
    # Khung chính
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Các góc
    corner_len = 25
    cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 255), 4)
    cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 255), 4)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), (0, 255, 255), 4)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), (0, 255, 255), 4)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), (0, 255, 255), 4)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), (0, 255, 255), 4)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (0, 255, 255), 4)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (0, 255, 255), 4)
    
    # Text hướng dẫn
    cv2.putText(frame, "Dat trai cay vao day", (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def predict_yolo(image):
    img_array = np.array(image)
    if model is not None:
        result = model(img_array, conf = 0.5, verbose = False)
        if len(result[0].boxes) > 0:
            det = result[0].boxes[0]
            conf = det.conf[0].cpu().numpy()
            return conf
def generate_frames():
    """Generator để stream video với YOLO detection"""
    global camera_active
    camera = get_camera()
    
    frame_count = 0
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        if model is not None:
            results = model(frame, conf=0.5, verbose=False)
            if len(results[0].boxes) > 0:
                det = results[0].boxes[0]
            annotated_frame = results[0].plot()
            
        else:
            annotated_frame = frame
        
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Camera Detection Active", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        frame_count += 1

@index_bp.route('/')
def index():
    return render_template('index.html')

@index_bp.route('/predict', methods=["POST"])
def predict():
    if 'fruit_image' not in request.files:
        return render_template('index.html', error="Không tìm thấy file ảnh!", result=None)

    file = request.files['fruit_image']

    if file.filename == '':
        return render_template('index.html', error="Chưa chọn file nào!", result=None)

    # Kiểm tra file hợp lệ
    ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXT:
        return render_template('index.html', error="Định dạng file không hợp lệ!", result=None)

    try:
        # ====== LƯU ẢNH ======
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # ====== YOLO PREDICT ======
        image = Image.open(save_path).convert('RGB')
        img_array = np.array(image)
        results = model(img_array, conf=0.5, verbose = False)
        res = results[0]

        # ====== KIỂM TRA NẾU KHÔNG CÓ DETECTION ======
        if res.boxes is None or len(res.boxes.cls) == 0:
            print(res.boxes)
            print("⚠ YOLO: No detections found.")
            return render_template('index.html',
                                   result=None,
                                   image_url=url_for('static', filename=f'uploads/{filename}'),
                                   error="Không phát hiện đối tượng nào!")

        # ====== LẤY KẾT QUẢ DETECTION ======

        det = res.boxes[0]
        cls = int(det.cls[0])
        best_class = model.names[cls]
        result_data = {
            "name": best_class.capitalize(),
            "image_url": url_for('static', filename=f'uploads/{filename}')
        }

        print("✔ Prediction:", result_data)

        return render_template('index.html', result=result_data)

    except Exception as e:
        print(f"❌ Error in predict: {e}")
        return render_template('index.html',
                               result=None,
                               error=f"Lỗi dự đoán: {e}")

@index_bp.route('/predict_camera', methods=["POST"])
def predict_camera():
    """Bật/tắt camera detection"""
    global camera_active
    
    action = request.json.get('action', 'start')
    
    if action == 'start':
        camera_active = True
        return jsonify({"status": "success", "message": "Camera started"})
    elif action == 'stop':
        camera_active = False
        release_camera()
        return jsonify({"status": "success", "message": "Camera stopped"})
    
    return jsonify({"status": "error", "message": "Invalid action"})

@index_bp.route('/video_feed')
def video_feed():
    """Stream video với YOLO detection"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@index_bp.route('/capture_frame', methods=["POST"])
def capture_frame():
    """Chụp ảnh từ camera và lưu kết quả"""
    try:
        camera = get_camera()
        success, frame = camera.read()
        
        if not success:
            return jsonify({"status": "error", "message": "Không thể đọc frame"})
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Lưu ảnh
        filename = f"camera_capture_{len(os.listdir(UPLOAD_FOLDER))}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(save_path, frame)
        
        # Run detection
        results = model(frame, conf=0.5, verbose=False)
        res = results[0]
        
        if res.boxes is None or len(res.boxes) == 0:
            return jsonify({
                "status": "no_detection",
                "message": "Không phát hiện trái cây nào",
                "image_url": url_for('static', filename=f'uploads/{filename}')
            })
        
        # Lấy detection tốt nhất
        best_idx = res.boxes.conf.argmax()
        best_class = model.names[int(res.boxes.cls[best_idx])]
        best_conf = float(res.boxes.conf[best_idx])
        
        return jsonify({
            "status": "success",
            "class": best_class,
            "confidence": f"{best_conf*100:.1f}%",
            "image_url": url_for('static', filename=f'uploads/{filename}')
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})