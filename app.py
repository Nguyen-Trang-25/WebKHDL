print("[1/5] Đang khởi động tiến trình...")
import os
import time

# Tắt cảnh báo oneDNN và các thông báo Info của TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ép TensorFlow chạy trên CPU (tránh lỗi treo khi khởi tạo card đồ họa GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("[2/5] Đang nạp các thư viện xử lý (OpenCV, Numpy, Joblib)...")
import cv2
import numpy as np
import joblib
print("[3/5] Đang nạp TensorFlow (Quá trình này có thể mất 15-30 giây, xin vui lòng kiên nhẫn)...")
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops
    from skimage.feature import local_binary_pattern

print("[4/5] Khởi tạo ứng dụng Flask Web...")
app = Flask(__name__)

# --- CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG CHO RANDOM FOREST ---
def extract_color_histogram(image_bgr, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], channels=[0, 1, 2], mask=None, histSize=list(bins), ranges=[0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

def extract_haralick_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.append(values.mean())
        features.append(values.std())
    return np.array(features, dtype=np.float32)

def extract_lbp_features(image_bgr, p=24, r=3):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=p, R=r, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist

def extract_hu_moments(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.astype(np.float32)

def extract_all_features(image_bgr, img_size):
    image_resized = cv2.resize(image_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    f_color = extract_color_histogram(image_resized)
    f_haralick = extract_haralick_features(image_resized)
    f_lbp = extract_lbp_features(image_resized)
    f_hu = extract_hu_moments(image_resized)
    features = np.concatenate([f_color, f_haralick, f_lbp, f_hu])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
# -------------------------------------------------------

# Config đường dẫn đến các file mô hình của nhóm
CNN_MODEL_PATH = "models/cnn_model.keras"
RF_PIPELINE_PATH = "models/random_forest_pipeline.pkl"

# Danh sách 8 loài lan đồng bộ với kết quả huấn luyện
CLASS_NAMES = [
    "Cattleya labiata", "Cymbidium lowianum", "Dendrobium anosmum",
    "Oncidium sphacelatum", "Paphiopedilum rothschildianum",
    "Phalaenopsis amabilis", "Rhynchostylis gigantea", "Vanda coerulea"
]

# Tải các mô hình vào bộ nhớ khi khởi động Server
print("=== Loading Models... ===")
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("-> CNN Model Loaded Successfully.")
except Exception as e:
    cnn_model = None
    print(f"-> [Warning] Cannot load CNN Model: {e}")

try:
    rf_pipeline = joblib.load(RF_PIPELINE_PATH)
    print("-> Random Forest Pipeline Loaded Successfully.")
except Exception as e:
    rf_pipeline = None
    print(f"-> [Warning] Cannot load Random Forest: {e}")
print("=========================")


# Route 1: Hiển thị giao diện Web chính
@app.route('/')
def index():
    return render_template('index.html')


# Route 2: API xử lý ảnh cho tính năng nhận diện đơn lẻ
@app.route('/predict_single', methods=['POST'])
def predict_single():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh gửi lên'}), 400
        
    file = request.files['file']
    model_type = request.form.get('model', 'cnn')
    
    if file.filename == '':
        return jsonify({'error': 'Chưa có ảnh nào được chọn'}), 400

    try:
        # Đọc ảnh từ request mà không cần lưu xuống ổ cứng (tối ưu bộ nhớ)
        img_pil = Image.open(file.stream).convert('RGB')
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        
        if model_type == 'cnn':
            if cnn_model is None:
                return jsonify({'error': 'Mô hình CNN chưa được nạp trên server'}), 500
            # Tiền xử lý theo cấu hình train_cnn.py của nhóm (224x224, scale 1/255)
            img_resized = cv2.resize(img_bgr, (224, 224))
            img_tensor = np.expand_dims(img_resized / 255.0, axis=0)
            
            preds = cnn_model.predict(img_tensor)
            class_idx = np.argmax(preds[0])
            confidence = float(preds[0][class_idx]) * 100
            predicted_class = CLASS_NAMES[class_idx]
            
        else:
            if rf_pipeline is None:
                return jsonify({'error': 'Mô hình Random Forest chưa được nạp trên server'}), 500
            # Tiền xử lý theo cấu hình train_rf.py của nhóm (Kích thước 128x128)
            rf_features = extract_all_features(img_bgr, 128)
            
            # Dự đoán thông qua Pipeline (Truyền đúng mảng đặc trưng đã trích xuất)
            preds_proba = rf_pipeline.predict_proba([rf_features])
            class_idx = np.argmax(preds_proba[0])
            confidence = float(preds_proba[0][class_idx]) * 100
            predicted_class = CLASS_NAMES[class_idx]
            
        inference_time = (time.time() - start_time) * 1000 # đổi sang miligiây
        
        return jsonify({
            'class_name': predicted_class,
            'confidence': f"{confidence:.2f} %",
            'time': f"{inference_time:.1f} ms"
        })
        
    except Exception as e:
        return jsonify({'error': f"Lỗi hệ thống trong quá trình xử lý: {str(e)}"}), 500


# Route 3: API xử lý đồng thời cả 2 mô hình (Phục vụ Tab Đối chứng)
@app.route('/predict_compare', methods=['POST'])
def predict_compare():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn ảnh'}), 400

    try:
        img_pil = Image.open(file.stream).convert('RGB')
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. Chạy nhanh nhánh CNN
        if cnn_model is not None:
            img_cnn = cv2.resize(img_bgr, (224, 224))
            tensor_cnn = np.expand_dims(img_cnn / 255.0, axis=0)
            preds_cnn = cnn_model.predict(tensor_cnn)
            idx_cnn = np.argmax(preds_cnn[0])
            cnn_result = {'class_name': CLASS_NAMES[idx_cnn], 'confidence': f"{preds_cnn[0][idx_cnn]*100:.2f} %"}
        else:
            cnn_result = {'class_name': 'Chưa load model', 'confidence': '0%'}
            
        # 2. Chạy nhanh nhánh Random Forest
        if rf_pipeline is not None:
            rf_features = extract_all_features(img_bgr, 128)
            preds_rf = rf_pipeline.predict_proba([rf_features])
            idx_rf = np.argmax(preds_rf[0])
            rf_result = {'class_name': CLASS_NAMES[idx_rf], 'confidence': f"{preds_rf[0][idx_rf]*100:.2f} %"}
        else:
            rf_result = {'class_name': 'Chưa load model', 'confidence': '0%'}
            
        return jsonify({
            'cnn': cnn_result,
            'rf': rf_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Đổi debug=False để tránh lỗi treo Terminal trên VS Code
    print("[5/5] SERVER ĐÃ SẴN SÀNG! Đang mở cổng mạng...")
    app.run(host='0.0.0.0', port=5001, debug=False)