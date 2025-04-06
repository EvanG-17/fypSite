from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pyrebase
from datetime import datetime
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # For using the softmax function
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model  # Import timm for EfficientNet model
import requests  # For downloading the model from Google Drive
from firebase_admin import auth, credentials

app = Flask(__name__)

load_dotenv()

config = {
  'apiKey': "AIzaSyBoUCKaswlxAlXTyO_5LCDjl10lEXqKmNg",
  'authDomain': "evanfypworking.firebaseapp.com",
  'projectId': "evanfypworking",
  'storageBucket': "evanfypworking.firebasestorage.app",
  'messagingSenderId': "413702763958",
  'appId': "1:413702763958:web:e4411b617ab80442f3bd17",
  'measurementId': "G-SLN4E8LJYN",
  'databaseURL': os.getenv("DATABASE_URL"),
}


firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
app.secret_key = os.getenv("SECRET_KEY")


@app.route('/login', methods=['GET', 'POST'])
def index():
    if 'user' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user_info = auth.get_account_info(user['idToken'])
            email_verified = user_info['users'][0]['emailVerified']

            if not email_verified:
                flash("Please verify your email before logging in.")
                return redirect(url_for('index'))

            session['user'] = email
            return redirect(url_for('home'))

        except:
            flash("Login failed. Please double-check your Username and Password.")
            return redirect(url_for('index'))

    return render_template('login.html', user=session.get('user'))




@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user_with_email_and_password(email, password)

            id_token = user['idToken']
            requests.post(
                "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=" + config['apiKey'],
                json={"requestType": "VERIFY_EMAIL", "idToken": id_token}
            )

            flash('Account created! Please check your email to verify before logging in.')
            return redirect(url_for('index'))

        except Exception as e:
            flash('Signup failed. Email may already be in use.')
            return redirect(url_for('signup'))

    return render_template('signup.html', user=session.get('user'))



# Delete all results
@app.route('/delete_results', methods=['POST'])
def delete_results():
    if 'user' not in session:
        return redirect(url_for('index'))

    safe_email = session['user'].replace('.', '_')
    firebase.database().child("results").child(safe_email).remove()

    flash("All results deleted successfully.")
    return redirect(url_for('results'))




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

# ✅ Load the model at app startup
print("Loading model into memory...")
model = create_model("efficientnet_b5", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded and ready.")


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

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('index'))

    user_email = session['user']
    safe_email = user_email.replace('.', '_')
    firebase_data = firebase.database().child("results").child(safe_email).get()

    results = []
    if firebase_data.each():
        for item in firebase_data.each():
            data = item.val()
            results.append({
                "date": data.get("date"),
                "image": data.get("image"),
                "label": data.get("label"),
                "probability": data.get("probability")
            })

    return render_template('results.html', results=results)



@app.route('/', methods=['GET', 'POST'])
def home():
    user_email = session.get('user')

    if request.method == 'POST':
        if 'fileUpload' not in request.files:
            return redirect(request.url)

        file = request.files['fileUpload']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = F.softmax(outputs, dim=1)
                deepfakeProbability = probabilities[0, 0].item()

            confidence = round(deepfakeProbability * 100, 2)
            
            # Set result based on 30% threshold
            result = "Deepfake" if confidence > 30 else "Real"

            # Only save results if logged in
            if user_email:
                safe_email = user_email.replace('.', '_')
                db = firebase.database()
                entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": file.filename,
                    "label": result,
                    "probability": confidence
                }
                db.child("results").child(safe_email).push(entry)

            return render_template(
                'home.html',
                uploaded_image=file.filename,
                result=result,
                probability=confidence,
                user=user_email
            )

    return render_template(
        'home.html',
        uploaded_image=None,
        result=None,
        probability=None,
        user=user_email
    )



@app.route('/account/delete')
def delete_account_page():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('delete_account.html')


@app.route('/google-login', methods=['POST'])
def google_login():
    data = request.get_json()
    id_token = data.get('idToken')

    if not id_token:
        return jsonify({"error": "Missing ID token"}), 400

    url = "https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=AIzaSyBoUCKaswlxAlXTyO_5LCDjl10lEXqKmNg"
    response = requests.post(url, json={"idToken": id_token})

    if response.status_code == 200:
        user_info = response.json()
        email = user_info['users'][0].get('email')

        if email:
            session['user'] = email
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Email missing"}), 400
    else:
        return jsonify({"error": "Invalid token or request"}), 401









 # App Route for privacy policy
@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy.html')


@app.context_processor
def inject_user():
    return dict(user=session.get('user'))


# Start Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Change port as required as Github LFS decides it does not like my FILE
    app.run(host="0.0.0.0", port=port)

