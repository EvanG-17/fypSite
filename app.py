from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # For using the softmax function
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model  # Import timm for EfficentNet model

# Firebase imports
import firebase_admin
from firebase_admin import auth, credentials
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# Firebase creds
cred = credentials.Certificate("config/firebase_credentials.json")
firebase_admin.initialize_app(cred)

# Create upload folder and allow specific file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL_PATH = "model/bestModel.pth"

model = create_model("efficientnet_b5", pretrained=False, num_classes=2)  # Ensure this is the same as the training setup
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # switch to evaluation mode when using model to predict

# make image same as training
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
            image = transform(image).unsqueeze(0).to(device)  # https://stackoverflow.com/questions/68824648/why-we-use-unsqueeze-function-while-image-processing
            
            
            with torch.no_grad(): # no_grad as training is not being done
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
    import os
    port = int(os.environ.get("PORT", 10000))  # Using renders own port due to "No open ports detected" error
    app.run(host="0.0.0.0", port=port, debug=True)

