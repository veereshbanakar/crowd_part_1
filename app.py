from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from DensedModel import CSRNet
from model2 import TensorHub
import base64
import io

app = Flask(__name__)

# Load the pretrained model
model = CSRNet()
checkpoint = torch.load('weights.pth', map_location="cpu")
model.load_state_dict(checkpoint)

tensor_hub_model = TensorHub()

@app.route('/')
def home():
    return render_template('home.html')

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Function to predict count and generate density map
def predict_count(img_path):
    # Load the image and preprocess it
    img = transform(Image.open(img_path).convert('RGB'))
    
    # Perform model inference
    output = model(img.unsqueeze(0))
    predicted_count = int(output.detach().cpu().sum().numpy())
    density_map = output.detach().cpu().numpy()[0, 0]
    density_map = density_map / np.max(density_map) * 255 
    
    return predicted_count, density_map.tolist()

# Define route for the home page
@app.route('/less_crowd')
def less_crowd():
    return render_template('index2.html')

@app.route('/densed_crowd')
def densed_crowd():
    return render_template('index1.html')

# Define route for processing image and returning results
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file has a valid extension
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to a temporary directory
        image_path = 'temp.jpg'
        file.save(image_path)
        
        # Call the predict_count function
        count, density_map = predict_count(image_path)
        
        # Convert density map to a NumPy array
        density_map_array = np.array(density_map)

        # Convert density map to a suitable mode (e.g., 'L' for grayscale) before saving it
        img = Image.fromarray(density_map_array.astype('uint8'))  # Convert to uint8 for saving
        
        # Encode the density map as base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        density_map_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Return the count and density map as JSON response
        return jsonify({'count': count, 'density_map': density_map_encoded})


@app.route('/predict1', methods=['POST'])
def predict1():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file has a valid extension
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Sreve the uploaded image to a temporary directory
        image_path = 'temp1.jpg'
        file.save(image_path)
        
        # Call the detect_persons method from TensorHub model
        count, annotated_image_path = tensor_hub_model.detect_persons(image_path)
        
        # Return the count and annotated image path as JSON response
        return jsonify({'count': count, 'annotated_image_path': annotated_image_path})

if __name__ == '__main__':
    app.run(debug=True)