import sys
sys.path.insert(1, 'lib')

from lib.OCR import ObjectCharacterRecognition
from lib.drive_scanner_runner import ModelRunner
from lib.db import OcrResult, OcrDatabase, Api
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from PIL import Image
import threading
from datetime import datetime
#from lib.OCR import OCR
from lib.drive_scanner_runner import ModelRunner

app = Flask(__name__)
# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)
db = OcrDatabase()

PATH_TO_TRAINED_MODEL = "lib/bow_updated_model"

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
def gen_frame():
    _, frame = camera.read()
    #get data that contain image, text, confidence
    data = ModelRunner.run([frame], PATH_TO_TRAINED_MODEL, ui=True)
    # cv2.imwrite('static/assets/capture.jpg', data[0])
    return data

            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/')
def capture_image():
    data = gen_frame()
    print(data)
    return render_template('ocr.html',data=data)

@app.route('/handle_data/', methods=['POST'])
def handle_data():
    pid = request.form['pid']
    model = request.form['model']
    serialNumber = request.form['serial']
    userReported = request.form['userReported']
    if userReported[0] == 'T':
        userReported = True
    else:
        userReported = False
    dt = datetime.now()
    ocr = OcrResult(int(pid), 0, '', model, '', serialNumber, userReported, dt).__dict__
    post = Api.insert(ocr, db.col)
    return render_template('index.html')

@app.route('/upload_image/', methods=['POST'])
def upload_image():
    data = request.files['file']
    img = Image.open(data)
    img = np.array(img)
    cv2.imwrite('static/assets/capture.jpg', img)
    data = ModelRunner.run([img], PATH_TO_TRAINED_MODEL, ui=True)
    return render_template('ocr.html',data=data)

if __name__ == "__main__":
    app.run(debug=True)
