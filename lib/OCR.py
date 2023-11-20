import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
import platform

# Line below required for Window users, (PATH variable issues for Windows specifically.)
if platform.system() == 'Windows':
	pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class ObjectCharacterRecognition:
	@staticmethod
	def green_blue_swap(image):
		# 3-channel image (no transparency)
		if image.shape[2] == 3:
			b, g, _ = cv2.split(image)
			image[:,:,0] = g
			image[:,:,1] = b
		# 4-channel image (with transparency)
		elif image.shape[2] == 4:
			b, g, _, _ = cv2.split(image)
			image[:,:,0] = g
			image[:,:,1] = b
		return image

	@staticmethod
	def getResults(image):
		"""The result of an image run through OCR with OCR settings of returning a dictionary of results.

		Args:
			image (_type_): _description_

		Returns:
			_type_: _description_
		"""
		# rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
		# plt.imshow(image)
		# plt.show()
		results = pytesseract.image_to_data(image, output_type=Output.DICT)
		return results

	@staticmethod
	def read(image, min_conf=0, psm=3, oem=3):
		# Convert image to RGB
		rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

		# PyTesseract settings
		custom_config = r'--oem {} --psm {}'.format(oem, psm)

		# Use Tesseract to localize each area of text in the input image
		results = pytesseract.image_to_data(rgb, config=custom_config, output_type=Output.DICT)

		# Initialize lists to store text and confidence levels
		list_text = []
		list_lv = []

		# Process each text localization
		for i in range(0, len(results["text"])):
			x = results["left"][i]
			y = results["top"][i]
			w = results["width"][i]
			h = results["height"][i]

			text = results["text"][i]
			conf = int(results["conf"][i])

			# Filter out weak confidence text localizations
			if conf > min_conf:
				print("Confidence: {}".format(conf))
				print("Text: {}".format(text))
				print("")

				list_text.append(text)
				list_lv.append(conf)

				# Draw bounding box and text
				text = "".join(text).strip()
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
				cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

		# Return a tuple that contains image, text list, and confidence list
		return image, list_text, list_lv
