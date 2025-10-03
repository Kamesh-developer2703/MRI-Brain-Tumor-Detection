from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import random
import sys
import pickle
from werkzeug.utils import secure_filename

import logging
import hashlib

# intialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# intialize flask app
app = Flask(__name__, static_folder='mri-images')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

CNN = None  # global variable to hold the CNN model

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global CNN
    if CNN is not None:
        return  # model already loaded
    script_dir = os.path.dirname(__file__)
    
    try:
        # Use JSON + weights method with compatibility fix
        model_json_path = os.path.join(script_dir, 'models', 'CNN_structure.json')
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        
        # Parse JSON and create model manually to avoid compatibility issues
        import json
        model_config = json.loads(model_json)
        
        # Create model from config
        CNN = tf.keras.Sequential()
        
        # Add layers one by one
        for layer_config in model_config['config']['layers']:
            layer_class = getattr(tf.keras.layers, layer_config['class_name'])
            layer_params = layer_config['config'].copy()
            
            # Handle InputLayer specially
            if layer_config['class_name'] == 'InputLayer':
                # Extract shape from batch_input_shape
                if 'batch_input_shape' in layer_params:
                    batch_input_shape = layer_params['batch_input_shape']
                    # Remove batch dimension to get input shape
                    input_shape = batch_input_shape[1:] if batch_input_shape[0] is None else batch_input_shape[1:]
                    layer_params['shape'] = input_shape
                    del layer_params['batch_input_shape']
            
            # Remove input_shape from Conv2D layers as it causes warnings
            elif 'batch_input_shape' in layer_params:
                del layer_params['batch_input_shape']
            
            layer = layer_class(**layer_params)
            CNN.add(layer)
        
        logger.info("Model structure created from JSON")

        # Load and set model weights
        weights_path = os.path.join(script_dir, 'models', 'CNN_weights.pkl')
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
            CNN.set_weights(weights)
        logger.info("Model weights loaded successfully")

        # Compile model
        CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        logger.info("Model loaded and compiled successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

# function for retrieving prediction from model given an image path
def get_model_prediction(image_path):
    load_model()
    try:
        # load and preprocess the image
        img = Image.open(image_path).resize((224, 224))
        # convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.expand_dims(np.array(img), axis=0)
        
        # predict using the CNN model
        prediction = CNN.predict(img_array)
        
        # interpret the prediction
        predicted_index = np.argmax(prediction[0])
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        predicted_class = class_labels[predicted_index]
        return predicted_class
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None

# load html template
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    try:  
         # select a random directory and then a random image within the image directory
        class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
        selected_class = random.choice(class_dirs)
        image_dir = os.path.join('mri-images', selected_class)
        image_name = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_name)
        predicted_label = get_model_prediction(image_path)
        web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')
        return jsonify({
            'image_path': web_accessible_image_path,
            'actual_label': selected_class,
            'predicted_label': predicted_label
        })
    except Exception as e:
        logger.error(f"Error in get-random-image route: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to make filename unique
        import time
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get prediction for the uploaded image
        predicted_label = get_model_prediction(file_path)
        
        if predicted_label is None:
            return jsonify({'error': 'Failed to process the image'}), 500
        
        # Return the result
        return jsonify({
            'image_path': f'/uploads/{filename}',
            'actual_label': 'Unknown (User Upload)',
            'predicted_label': predicted_label,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error in upload-image route: {e}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
