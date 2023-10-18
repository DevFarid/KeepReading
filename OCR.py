import pytesseract
from pytesseract import Output
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class OCR:
	def __init__(self) -> None:
		pass

	@staticmethod
	def green_blue_swap(image):
		# 3-channel image (no transparency)
		if image.shape[2] == 3:
			b,g,r = cv2.split(image)
			image[:,:,0] = g
			image[:,:,1] = b
		# 4-channel image (with transparency)
		elif image.shape[2] == 4:
			b,g,r,a = cv2.split(image)
			image[:,:,0] = g
			image[:,:,1] = b
		return image

	@staticmethod
	def read(image, min_conf=0):
		# We load the input image and then convert
		# it to RGB from BGR. We then use Tesseract
		# to localize each area of text in the input
		# image
		rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
		results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

		# Then loop over each of the individual text
		# localizations
		for i in range(0, len(results["text"])):
			
			# We can then extract the bounding box coordinates
			# of the text region from the current result
			x = results["left"][i]
			y = results["top"][i]
			w = results["width"][i]
			h = results["height"][i]
			
			# We will also extract the OCR text itself along
			# with the confidence of the text localization
			text = results["text"][i]
			conf = int(results["conf"][i])
			
			# filter out weak confidence text localizations
			if conf > min_conf:
				
				# We will display the confidence and text to
				# our terminal
				print("Confidence: {}".format(conf))
				print("Text: {}".format(text))
				print("")
				
				# We then strip out non-ASCII text so we can
				# draw the text on the image We will be using
				# OpenCV, then draw a bounding box around the
				# text along with the text itself
				text = "".join(text).strip()
				cv2.rectangle(image,
							(x, y),
							(x + w, y + h),
							(0, 0, 255), 2)
				cv2.putText(image,
							text,
							(x, y - 10),
							cv2.FONT_HERSHEY_SIMPLEX,
							1.2, (0, 255, 255), 3)
		return image
		# After all, we will show the output image
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)
    