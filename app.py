from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = tf.keras.models.load_model('./my_image_classification_model.h5')
class_names = ['Chrysanthemum', 'Hibiscus', 'Marigold', 'Petunia', 'Rose']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            class_name, confidence = predict_image(filepath)
            if confidence < 0.5:  # Adjust threshold as needed
                result = "No matching flower class found"
            else:
                result = f"Predicted: {class_name} (Confidence: {confidence:.2%})"
            return render_template('index.html', 
                                 result=result, 
                                 image_path=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)