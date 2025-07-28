import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, template_folder='tamplate') # Corrected 'tamplate' to 'template'

# Load the model
model = load_model('C:/Users/darsh/OneDrive/Documents/agriculter_project/plant_desis_detection/model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery Mildew', 2: 'Anthracnose'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Ensure the uploads directory exists
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)  # Create uploads directory if it doesn't exist

        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)
        
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return render_template('result.html', prediction=predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)



