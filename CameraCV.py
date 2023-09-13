import cv2

class CameraCV:
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
        # 1.creating a video object
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera


        # 4.Create a frame object
        check, frame = video.read()
        # Converting to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 5.show the frame!
        cv2.imshow("Capturing", frame)
    
        # 7. image saving
        showPic = cv2.imwrite("snapshot.jpg", frame)
        print(showPic)

        # 8. shutdown the camera
        video.release()
        cv2.destroyAllWindows()