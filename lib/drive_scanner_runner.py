from drive_scanner_updated import *

import pytesseract
from pytesseract import Output

import argparse
import cv2
import os
import platform

from queue import Queue
from threading import Thread, Lock
from CropInfoReader import CropInfoReader

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_PID_only(file_name):
    return file_name[:file_name.find(".jpg")]


def remove_suffix(word: str, removals: list):
    for removal in removals:
        if word.endswith(removal):
            return word[:len(word) - len(removal)], removal

class ModelRunner():

    @staticmethod
    def run(arg_image: list, arg_training_data_path: str, arg_batch=False, arg_extension=".jpg", arg_threads=10, arg_results="", arg_accuracy=False, arg_exclusions="", ui=False):
        crop_info = CropInfoReader.getCropInfo()
        if ui:
            ConstantFilePaths(True)
        if not arg_batch:
            # pre-process image
            arg_image = [Preprocess.crop_background(image) for image in arg_image]
            # load image
            im_and_text = [(image, pytesseract.image_to_data(image, output_type=Output.DICT)['text']) for image in arg_image]
            PIDs = []
            SNs = []
            MODs = []
            
            for entry in im_and_text:
                # getPID
                PIDs.append(getPID(entry[1], entry[0]))
                # getMOD
                modelN = getMOD(entry[0], ConstantNames.DRIVES, entry[1], arg_training_data_path)
                modelN, manufacturer = remove_suffix(modelN, ConstantNames.DRIVES)
                MODs.append(modelN)
                # Remove manufactorer at the end of modelN
                # modelN = str(modelN)
                # for i in DRIVES:
                #     if i in modelN:
                #         modelN = modelN.replace(i, "")
                cropped_image = Preprocess.crop_to_ser_no(entry[0], str(modelN), crop_info)
                # getSN
                SNs.append(getSER(cropped_image, modelN))

            return [{"PID": PIDs[0], "SN": SNs[0], "MN": MODs[0]}]
        else:

            d_queue = Queue()
            r_queue = Queue()

            for image in arg_image:
                d_queue.put(image)

            for _ in range(arg_threads):
                d_queue.put(None)

            def process_image(image):
                results = pytesseract.image_to_string(image, output_type=Output.DICT)
                return results.copy()
            
            def crop_image_based_on_MOD(image, modelN):
                modelN = str(modelN)
                for i in ConstantNames.DRIVES:
                    if i in modelN:
                        modelN = modelN.replace(i, "")
                return Preprocess.crop_to_ser_no(image, modelN, crop_info)
                
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
                        modelNumber = getMOD(image, ConstantNames.DRIVES, ocr_text, arg_training_data_path, arg_accuracy)

                            
                    modelNumber, manufacturer = remove_suffix(modelNumber, ConstantNames.DRIVES)
                    cropped_image = crop_image_based_on_MOD(image, modelNumber)
                    results = [image_loc, getPID(ocr_text, image), getSER(cropped_image, modelNumber), modelNumber]

                    r_queue.put(results.copy())
                    d_queue.task_done()
            
            threads = []
            for _ in range(arg_threads):
                thread = Thread(target=image_processing_worker)
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            processed_results = []
            while not r_queue.empty():
                results = r_queue.get()
                processed_results.append(results)

            final_list = []
            for result in processed_results:
                final_list.append({"drive_label": result[0], "PID": result[1][1], "SN": result[2][1], "MN": result[3]})
            return final_list

if __name__ == "__main__":
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
    ap.add_argument("--exclusions",
                    help="images to exclude",
                    default="")

    args = vars(ap.parse_args())

    # case with single image
    if not args['batch']:
        if "lib" in args['trained_data']:
            ConstantFilePaths(True)
        arg_image = [cv2.imread("..\\" + args['image'])]
        args_training_data = args['trained_data']
        args_exclusions = args['exclusions']
        args_extension = args['extension']
        args_batch = False
        args_threads = 10

    # case with batch images
    else:
        all_files = os.listdir(args['image'])
        exclusions = args['exclusions'].split(',')[:-1]
        selected_files = [os.path.join(args['image'], file) for file in all_files if (file.endswith(args['extension']) and get_PID_only(file) not in exclusions)]

        arg_image = [(cv2.imread(image_loc), image_loc) for image_loc in selected_files]

        args_training_data = args['trained_data']
        args_exclusions = args['exclusions']
        args_extension = args['extension']
        args_batch = True
        args_threads = args['threads']

    model_results = ModelRunner.run(arg_image, args_training_data, arg_batch=args_batch, arg_threads=args_threads, arg_extension=args_extension, arg_exclusions=args_exclusions)

    if args['batch']:
        for i, result in enumerate(model_results):
            with open(args["results"], "a") as res_file:
                """
                res_file.write(f'\tDrive Number: {i}')
                res_file.write(f'\tPID: {result["PID"]}')
                res_file.write(f'\tSerial Number: {result["SN"]}')
                res_file.write(f'\tModel Number: {result["MN"]}')
                res_file.write('\n')
                """
                res_file.write(f'{result["drive_label"]},{result["PID"]},{result["SN"]},{result["MN"]}')
                res_file.write("\n")
    else:
        for result in model_results:
            with open(args["results"], "w") as res_file:
                res_file.write(f'\tPID: {result["PID"]}')
                res_file.write(f'\tSerial Number: {result["SN"]}')
                res_file.write(f'\tModel Number: {result["MN"]}')
                res_file.write('\n')


