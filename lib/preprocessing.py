import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import label
from scipy import ndimage
import cv2
from rembg import remove
from skimage.filters import threshold_otsu
from barcode_detection import BarcodeDetection

class Preprocess():

    def to_bw(image: ArrayLike) -> ArrayLike:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def to_binary(image: ArrayLike, threshold: int) -> ArrayLike:
        return np.where(image > threshold, 1, 0)
    
    def remove_bg(image: ArrayLike) -> ArrayLike:
        return remove(image)
    
    def gaussian_blur(image: ArrayLike, sigma: float) -> ArrayLike:
        return ndimage.gaussian_filter(image, sigma)
    
    def remove_bg_to_bw_blur_binary(img, threshold=0, sigma=1):
        threshold = threshold_otsu(Preprocess.to_bw(img))
        return Preprocess.to_bw(img)
       
    def resize(img):
        RESIZE_WIDTH = 1000
        RESIZE_HEIGHT = 1200
        return cv2.resize(img, (RESIZE_WIDTH,RESIZE_HEIGHT))

    def crop_background(img):
        L,_ = label(Preprocess.remove_bg(img))
        mask = (L!=L[0,0]).any(-1)
        out = img[np.ix_(mask.any(1), mask.any(0))]
        return Preprocess.resize(out)

    def crop_to_ser_no(img, model, cropInfo):
        ##Crop image to where the serial number is located
        def crop_to_revelant_info():
            if model in cropInfo:
                topY, bottomY, leftX, rightX = cropInfo[model]
                return img[topY:bottomY, leftX:rightX]
            return img
        img = Preprocess.crop_background(img)
        img = Preprocess.resize(img)
        img = Preprocess.remove_bg_to_bw_blur_binary(img)
        img = crop_to_revelant_info()
        img = BarcodeDetection.detect(img)
        return img
        