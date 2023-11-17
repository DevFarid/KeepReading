from drive_scanner_updated import *

import pytesseract
from pytesseract import Output

import argparse
import cv2
import os
import platform

from queue import Queue
from threading import Thread, Lock

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
                default="model")
ap.add_argument("-r", "--results",
                help="file to record scanned PIDs, Serial Nums, and Model Nums",
                default="scan_results.txt")
ap.add_argument("--threads",
                help="number of threads to use if batch processing",
                default=10)
ap.add_argument("--accuracy",
                help="turns on \'accuracy\' mode to test accuracy of model",
                default=False)

args = vars(ap.parse_args())

DRIVES = ["DELL", "HP", "SEAGATE", "HP", "SAMSUNG", "HGST", "LENOVO"]
if not args['batch']:
    # load image
    images = [cv2.imread("..\\" + args['image'])]
    im_and_text = [(image, pytesseract.image_to_data(image, output_type=Output.DICT)['text']) for image in images]
    # getPID
    PIDs = [getPID(entry[1], entry[0]) for entry in im_and_text]

    # getSN
    SNs = [getSER(entry[1], entry[0]) for entry in im_and_text]

    # getMOD
    MODs = [getMOD(entry[0], DRIVES, entry[1], args["trained_data"]) for entry in im_and_text]

    for i in range(len(im_and_text)):
        with open(args["results"], "w") as res_file:
            res_file.write(f'DRIVE NUM: {i}')
            res_file.write(f'\tPID: {PIDs[i]}')
            res_file.write(f'\tSerial Number: {SNs[i]}')
            res_file.write(f'\tModel Number: {MODs[i]}')
            res_file.write('\n')

    print("Complete.")

else:
    all_files = os.listdir(args['image'])
    selected_files = [os.path.join(args['image'], file) for file in all_files if file.endswith(args['extension'])]

    images = [(cv2.imread(image_loc), image_loc) for image_loc in selected_files]
    d_queue = Queue()
    r_queue = Queue()

    for image in images:
        d_queue.put(image)

    for _ in range(args['threads']):
        d_queue.put(None)

    def process_image(image):
        results = pytesseract.image_to_string(image, output_type=Output.DICT)
        return results.copy()
    
    MUTEX = Lock()
    def image_processing_worker():
        while True:
            image_info = d_queue.get()
            if image_info is None:
                break
            image = image_info[0]
            image_loc = image_info[1]
            ocr_text = process_image(image)

            with MUTEX:
                modelNumber = getMOD(image, DRIVES, ocr_text, args['trained_data'], args['accuracy'])

            results = [image_loc, getPID(ocr_text, image), getSER(ocr_text, image), modelNumber]

            r_queue.put(results.copy())
            d_queue.task_done()
    
    threads = []
    for _ in range(args['threads']):
        thread = Thread(target=image_processing_worker)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    processed_results = []
    while not r_queue.empty():
        results = r_queue.get()
        processed_results.append(results)

    for i, result in enumerate(processed_results):
        with open(args["results"], "a") as res_file:
            """
            res_file.write(f'DRIVE NUM: {i + 1}')
            res_file.write(f'\tPID: {result[0]}')
            res_file.write(f'\tSerial Number: {result[1]}')
            res_file.write(f'\tModel Number: {result[2]}')
            """
            res_file.write(f'{result[0]},{result[1][1]},{result[2][1]},{result[3]}')
            res_file.write('\n')



