from drive_scanner_updated import *

import pytesseract
from pytesseract import Output

import argparse
import cv2
import os
import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                required=True,
                help="path to image to scan")
ap.add_argument("-b", "--batch",
                type=bool,
                help="if true, arg -i will be read as a folder containing many images",
                default=False)
ap.add_argument("-e", "--extension",
                help="file extension to load in if -b is True",
                default=".jpg")
ap.add_argument("-t", "--trained_data",
                help="location where saved model is kept",
                default=["Model\\trained_means.yaml"],
                nargs='+')
ap.add_argument("-r", "--results",
                help="file to record scanned PIDs, Serial Nums, and Model Nums",
                default="Model\\scan_results.txt")
args = vars(ap.parse_args())

if not args['batch']:
    # load image
    images = [cv2.imread(args['image'])]
else:
    all_files = os.listdir(args['image'])
    selected_files = [os.path.join(args['image'], file) for file in all_files if file.endswith(args['extension'])]

    images = [cv2.imread(image_loc) for image_loc in selected_files]


# get text from image
im_and_text = [(image, pytesseract.image_to_data(image, output_type=Output.DICT)['text']) for image in images]

# getPID
PIDs = [getPID(entry[1], entry[0]) for entry in im_and_text]

# getSN
SNs = [getSER(entry[1], entry[0]) for entry in im_and_text]

# getMOD
drives = ["DELL", "HP", "SEAGATE", "HP", "SAMSUNG", "HGST", "LENOVO"]
MODs = [getMOD(entry[0], drives, entry[1], args["trained_data"]) for entry in im_and_text]

for i in range(len(im_and_text)):
    with open(args["results"], "w") as res_file:
        res_file.write(f'DRIVE NUM: {i}')
        res_file.write(f'\tPID: {PIDs[i]}')
        res_file.write(f'\tSerial Number: {SNs[i]}')
        res_file.write(f'\tModel Number: {MODs[i]}')

print("Complete.")