import cv2
import zxingcpp
import numpy
import pytesseract
from PIL import Image, ImageTk

class CameraCV:

    global video
    video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

    """
    CameraCV is a utility class providing methods and access to user's camera device.
    Created by Farid on 09/13/2023.
    """
    @staticmethod
    def captureCameraPicture():
        """
        Creates a window GUI with the user's camera (if applicable) as a preview.
        Upon pressing 'q', it snaps a picture called `snapshot.jpg`, then exits
        """        

        # 1.Create a frame object
        check, frame = video.read()
        # Converting to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 4.show the frame!
        cv2.imshow("Capturing", frame)
    
        # 5. image saving
        showPic = cv2.imwrite("snapshot.jpg", frame)

    def captureCameraPreview():
        # 1.Create frame object
        _, frame = video.read()

        # 2.Create tkinter friendly image
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        img_update = ImageTk.PhotoImage(Image.fromarray(frame))

        return img_update
    
    @staticmethod
    def removeBarCodeFromImage(image):
        """
        Modifies the image by blocking the barcode.
        """
        results = zxingcpp.read_barcodes(image)
        for result in results:
            t = str(result.position)[:-1]
            t = [list(map(int, x.split("x"))) for x in t.split(" ")]
            
            coords = {
                "top_right": {
                    "x": t[0][0],
                    "y": t[0][1],
                },
                "bottom_right": {
                    "x": t[1][0],
                    "y": t[1][1],
                },
                "bottom_left": {
                    "x": t[2][0],
                    "y": t[2][1],
                },
                "top_left": {
                    "x": t[3][0],
                    "y": t[3][1],
                },
            }
            # Use coordinates of barcodes to cover them up with rectangles, improving OCR results
            image = cv2.rectangle(
                image, (coords["top_left"]["x"],coords["top_left"]["y"]),(coords["bottom_right"]["x"],coords["bottom_right"]["y"]), (0, 0, 255), -1)
        return image