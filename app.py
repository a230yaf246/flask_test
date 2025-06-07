from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'eye_disease_classifier_model.h5'
model = load_model(MODEL_PATH)

label_map = {
    0: "Cataract",
    1: "Conjunctivitis",
    2: "Eyelid",
    3: "Normal",
    4: "Uveitis"
}

IMG_SIZE = (128, 128)

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def index():
    return jsonify({'message': 'API is running.'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    files = request.files.getlist('image')

    results = []

    for file in files:
        try:
            image = preprocess_image(file)
            prediction = model.predict(image)
            pred_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            label = label_map[pred_index]

            results.append({
                'filename': file.filename,
                'class': label,
                'confidence': f"{confidence:.2%}"
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
