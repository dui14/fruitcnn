from flask import Blueprint, render_template, request, url_for
from app.utils.file_utils import FileUtils
from app.services.predict import Predict
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

MODEL_PATH = 'fruit_model_full.h5'
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

index_bp = Blueprint('index', __name__)

@index_bp.route('/')
def index():
    return render_template('index.html')


@index_bp.route('/predict', methods=["POST"])
def predict():
    if 'fruit_image' not in request.files:
        return render_template('index.html', result=None, error="Không tìm thấy file ảnh!")

    file = request.files['fruit_image']

    if file.filename == '':
        return render_template('index.html', result=None, error="Chưa chọn file nào!")

    if not FileUtils.allowed_file(file.filename):
        return render_template('index.html', result=None, error="Định dạng file không hợp lệ!")

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        img_array = FileUtils.preprocess(save_path)
        if img_array is None:
            return render_template('index.html', result=None, error="Lỗi khi xử lý ảnh!")

        result = Predict.predict(model, img_array)

        result["image_url"] = url_for('static', filename=f'uploads/{filename}')

        print(f" Prediction result: {result}")
        return render_template('index.html', result=result)

    except Exception as e:
        print(f" Error in predict: {e}")
        return render_template('index.html', result=None, error=f"Error in predict: {e}")
