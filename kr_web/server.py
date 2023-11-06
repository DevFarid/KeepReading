from flask import Flask, render_template, Response
import cv2
import threading
from kr_utils import OCR

class web_server():
    app = Flask(__name__)
    def __init__(self) -> None:

        self.camera = cv2.VideoCapture(0)

    def gen_frames(self):  
        while True:
            success, frame = self.camera.read()  # read the camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                
    def gen_frame(self):
        _, frame = self.camera.read()
        #get data that contain image, text, confidence
        data = OCR.read(frame)
        cv2.imwrite('static/assets/capture.jpg', data[0])
        return data
                
    @app.route('/')
    def index(self):
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed(self):
        return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/capture/')
    def capture_image(self):
        data = self.gen_frame()
        return render_template('ocr.html')

if __name__ == "__main__":
    web_server.app.run(debug=True)