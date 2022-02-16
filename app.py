import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
# from werkzeug import secure_filename
from werkzeug.serving import run_simple

# Keras
from keras.models import load_model

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/', methods=['POST'])
def index():
  if 'file' not in request.files:
    return 500

  # Read file
  img = np.fromfile(request.files.get('file'), np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (28, 28))

  # Convert file to array
  gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
  gray = gray.reshape(1, 28, 28, 1)
  gray /= 255

  prediction = model.predict(gray)
  prediction = prediction.argmax()

  return {
    "prediction": int(prediction)
  }

if __name__ == '__main__':
  model = load_model('./model.h5')
  run_simple('0.0.0.0', 5000, app, use_reloader=True)
