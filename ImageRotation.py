import cv2
import pytesseract
import imutils
from pytesseract import Output

class ImageRotation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def rotate(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        # display the orientation information
        print("[INFO] ".format(results.keys()))
        print("[INFO] detected orientation: {}".format(results["orientation"]))
        print("[INFO] rotate by {} degrees to correct".format(results["rotate"]))

        if results["orientation"] != results["rotate"]:
            image = imutils.rotate_bound(image, angle=results["rotate"])
        return image