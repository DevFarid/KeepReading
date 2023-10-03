import cv2
from PIL import Image, ImageTk
from OCR import OCR

class CameraCV:
    """
    CameraCV is a utility class providing methods and access to user's camera device.
    Created by Farid on 09/13/2023.
    """
    def __init__(self) -> None:
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

    def analyze(self):
        img = self.captureCameraPicture()
        OCR.read(img)

    def captureCameraPicture(self):
        _, frame = self.video.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("snapshot.jpg", frame)
        return cv2.imread("snapshot.jpg")
        

    def captureCameraPreview(self):
        _, frame = self.video.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        return ImageTk.PhotoImage(Image.fromarray(frame))