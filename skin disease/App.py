from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename  # To safely handle file names

# Initialize Flask app with static folder for images
app = Flask(__name__, static_folder='static')

# Load the saved Keras model
model = load_model('save_model.keras')

# Define the disease names (make sure the names match the training labels)
disease_names = ['Chickenpox', 'Eczema', 'Ringworm', 'Disease4', 'Disease5', 'Disease6', 'Disease7', 'Chickenpox', 'Disease9']

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file!", 400

    # Save the file to a temporary location using secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save(filepath)

    # Load and preprocess the image
    image = cv2.imread(filepath)
    image = cv2.resize(image, (256, 256))  # Resize to match the input size of the model
    image = image / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    predicted_disease = disease_names[predicted_class]

    # Optional: Remove the image file after prediction to avoid clutter
    os.remove(filepath)

    # Pass the predicted disease to the frontend
    return render_template('result.html', predicted_disease=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
