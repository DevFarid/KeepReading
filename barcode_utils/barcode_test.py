from BarcodeDetection import BarcodeDetection
import cv2


image = BarcodeDetection.detect(cv2.imread("4421178.jpg"))

cv2.imshow("Barcode Removal Example", cv2.resize(image, (500,500), interpolation=cv2.INTER_AREA))
cv2.waitKey(0)
