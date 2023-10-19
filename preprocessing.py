import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage
import cv2
from rembg import remove

class Preprocess():

    def to_bw(image: ArrayLike) -> ArrayLike:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def to_binary(image: ArrayLike, threshold: int) -> ArrayLike:
        return np.where(image > threshold, 1, 0)
    
    def remove_bg(image: ArrayLike) -> ArrayLike:
        return remove(image)
    
    def gaussian_blur(image: ArrayLike, sigma: float) -> ArrayLike:
        return ndimage.gaussian_filter(image, sigma)