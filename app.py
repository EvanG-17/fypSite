from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # For using the softmax function
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model  # Import timm for EfficientNet model
import requests  # For downloading the model from Google Drive

app = Flask(__name__)

# Create upload folder and allow specific file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path and Google Drive file ID
MODEL_PATH = "model/bestModel.pth"
FILE_ID = os.getenv("GDRIVE_FILE_ID")  # Google drive ID file for model

# Function to download the model from Google Drive
def download_model_from_gdrive(file_id, save_path):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure model directory exists

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
    
    print("Model downloaded successfully!")

# Check if model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    download_model_from_gdrive(FILE_ID, MODEL_PATH)

# Load the model
model = create_model("efficientnet_b5", pretrained=False, num_classes=2)  # Ensure this is the same as the training setup
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Switch to evaluation mode when using the model to predict

# Make image same as training
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Allowed file types. You can add more if needed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'fileUpload' not in request.files:
            return redirect(request.url)

        file = request.files['fileUpload']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Open image for processing
            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)  # Convert to tensor
            
            with torch.no_grad():  # No_grad as training is not being done
                outputs = model(image)  # Get logits
                probabilities = F.softmax(outputs, dim=1)  # Convert our logits to probabilities
                prediction = torch.argmax(outputs, dim=1).item() 
                deepfakeProbability = probabilities[0, 0].item()  # Get confidence score for Deepfake class

            confidence = round(deepfakeProbability * 100, 2)  # Convert to percentage
            result = "Deepfake"  # Always say Deepfake but use score

            return render_template('home.html', uploaded_image=file.filename, result=result, probability=confidence)

    return render_template('home.html', uploaded_image=None, result=None, probability=None)

# Start Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get Render-assigned port
    app.run(host="0.0.0.0", port=port, debug=True)

