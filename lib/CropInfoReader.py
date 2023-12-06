import os
class CropInfoReader:
	def __init__(self) -> None:
		pass

	@staticmethod
	def checkLib(s):
		if "lib" in os.getcwd():
			if "lib/" in s[:4]:
				return s[4:]
			else:
				return s
		else:
			if "lib/" in s[:4]:
				return s
			else:
				return "lib/" + s[:4]
	
	@staticmethod
	def getCropInfo():
		cropDimension = {}
		# get crop-dimensions from txt file

		with open(CropInfoReader.checkLib('lib/crop_details.txt'),"r") as file:
			lines = file.readlines()
			model = ""
			dimensions = ""
			for s in lines:
				if s[0].isalpha():
					model = s.strip('\n')
					dimensions = ""
				elif s[0] == '\n':
					continue
				else:
					dimensions = s
					
				if dimensions:
					list = s.split(',')
					list = [int(n) for n in list]
					cropDimension[model] = list
		return cropDimension