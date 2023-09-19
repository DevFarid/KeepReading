import cv2
import zxingcpp
import numpy
import pytesseract

image = cv2.imread('data/4421175.jpg')
# read_barcodes: Reads and translates barcodes, but here it is just used for positional data
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
    print(coords["top_left"]["y"])
    # Use coordinates of barcodes to cover them up with rectangles, improving OCR results
    image = cv2.rectangle(
		image, (coords["top_left"]["x"],coords["top_left"]["y"]),(coords["bottom_right"]["x"],coords["bottom_right"]["y"]), (0, 0, 255), -1)
    cv2.imshow("rectangle", cv2.resize(image, (500,500), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
print(pytesseract.image_to_string(image))
