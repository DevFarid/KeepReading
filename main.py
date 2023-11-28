import sys
sys.path.insert(1, 'lib')

from lib.OCR import ObjectCharacterRecognition
from lib.drive_scanner_runner import ModelRunner
from flask import Flask, render_template, Response
import cv2
import threading
import json
#from lib.OCR import OCR
from lib.drive_scanner_runner import ModelRunner

app = Flask(__name__)
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
    data = ModelRunner.run([frame], "lib/all_training", ui=True)
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

if __name__ == "__main__":
    app.run(debug=True)
