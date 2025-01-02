from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# create upload folder and allow the following extensions:
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# flask app uses upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ensures static/uploads exists and if not it makes it.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# we split filename by last period in it (.)
def allowed_file(filename):
    # extracts and converts to lowercase
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    # here is file upload logic:
    if request.method == 'POST':
        # check if file is uploaded
        if 'fileUpload' not in request.files:
            return redirect(request.url)

        # stores uploaded files
        file = request.files['fileUpload']

        # If no file is selected
        if file.filename == '':
            return redirect(request.url)

        # Save the file if it's allowed
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return render_template('home.html', uploaded_image=file.filename)

    # no image displayed when user visits our home page
    return render_template('home.html', uploaded_image=None)

# starts Flask application in debug mode, this helps display error messages in browser which are more detailed than usual.
if __name__ == '__main__':
    app.run(debug=True)
