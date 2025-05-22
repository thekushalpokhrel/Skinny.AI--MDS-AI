import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from datetime import datetime
from utils import preprocess_image, create_model, train_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = [
    'akiec',  # Actinic keratoses and intraepithelial carcinoma
    'bcc',    # Basal cell carcinoma
    'bkl',    # Benign keratosis-like lesions
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic nevi
    'vasc'    # Vascular lesions
]
MODEL_PATH = 'static/models/skin_disease_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/models', exist_ok=True)

# Load or initialize model
try:
    model = load_model(MODEL_PATH)
except:
    print("Pre-trained model not found. Creating a new model...")
    model = create_model(len(CLASSES))
    model.save(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html', classes=CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        img = preprocess_image(filepath)
        predictions = model.predict(img)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'filename': filename
        })

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Get parameters from request
        epochs = int(request.form.get('epochs', 5))
        batch_size = int(request.form.get('batch_size', 32))

        # Verify dataset exists
        train_dir = os.path.join('dataset', 'train')
        if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
            return jsonify({'status': 'error', 'message': 'No training data found'}), 400

        # Train the model
        history = train_model(model, CLASSES, epochs, batch_size)

        # Save the updated model
        model.save(MODEL_PATH)

        return jsonify({
            'status': 'success',
            'accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/add_to_dataset', methods=['POST'])
def add_to_dataset():
    if 'datasetFiles' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('datasetFiles')
    class_name = request.form.get('class_name')
    is_test = request.form.get('is_test') == 'on'

    if not class_name or class_name not in CLASSES:
        return jsonify({'error': 'Invalid class name'}), 400

    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400

    saved_count = 0
    dataset_type = 'test' if is_test else 'train'

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            class_dir = os.path.join('dataset', dataset_type, class_name)
            os.makedirs(class_dir, exist_ok=True)
            filepath = os.path.join(class_dir, filename)

            # Check if file already exists
            if not os.path.exists(filepath):
                file.save(filepath)
                saved_count += 1

    return jsonify({
        'status': 'success',
        'added_count': saved_count,
        'class_name': class_name,
        'dataset_type': dataset_type
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)