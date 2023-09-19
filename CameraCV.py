import cv2
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