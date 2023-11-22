import cv2
from PIL import Image, ImageTk
from OCR import OCR
# from BarcodeDetection import BarcodeDetection

class CameraCV:
    """
    CameraCV is a utility class providing methods and access to user's camera device.
    Created by Farid on 09/13/2023.
    """
    def __init__(self) -> None:
        # self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def analyze(self):
        img = self.captureCameraPicture()
        img = OCR.read(img)

    def captureCameraPicture(self):
        _, frame = self.video.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("snapshot.jpg", frame)
        return cv2.imread("snapshot.jpg")
        

    def captureCameraPreview(self, w, h):
        _, frame = self.video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        # Original dimensions
        original_h, original_w = frame.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = original_w / original_h
        
        # Calculate new dimensions based on aspect ratio
        if w / h > aspect_ratio:
            new_w = int(h * aspect_ratio)
            new_h = h
        else:
            new_h = int(w / aspect_ratio)
            new_w = w

        # Resize frame while keeping aspect ratio
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return ImageTk.PhotoImage(Image.fromarray(resized_frame))

