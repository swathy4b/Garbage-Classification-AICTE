import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = load_model('best_garbage_model.h5')
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Print model summary for debugging
print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Read the image file
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'error': 'Empty file'}), 400
            
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if image is RGBA or grayscale
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess the image
        img = img.resize((224, 224))  # Resize to match MobileNetV2's expected input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        # Check if image has the right number of channels
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all predictions with their confidence scores
        all_predictions = {
            class_name: float(confidence) 
            for class_name, confidence in zip(classes, predictions[0])
        }
        
        return jsonify({
            'class_name': classes[predicted_class_idx],
            'class_index': int(predicted_class_idx),
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in classification: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to process image',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
