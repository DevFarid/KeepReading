# Tesseract and image processing
import argparse
import cv2
from pytesseract import *
import pytesseract
import PIL

# File processing
import csv
import yaml
import matplotlib.pyplot as plt

# Import models + preprocessors
from models_updated import CModel, KNearest
from preprocessing import *
from data_representation_abstracted_updated import TrainingRepresentation, BWHistogram

# Threading
from queue import *
from threading import Thread

# Set pytesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
"""
# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                required=True,
                help="path to folder with image to scan")
args = vars(ap.parse_args())

# load image
image = cv2.imread(args['image'])

# get text from image
scanned_text = pytesseract.image_to_data(image, output_type=Output.DICT)
print(scanned_text['text'])
"""
# load models
def load_models(trained_data: str):
    model_params = []

    k_model = KNearest()
    k_model.train(trained_data)
    parameters = {'K': 1, 'optimal_size': len(k_model._KNearest__training[0][1])}
    model_params.append([k_model, parameters.copy()])

    return model_params

# predict based on model
def predict(image, model: CModel, data_rep: TrainingRepresentation, parameters: dict):
    prediction = model.predict(image, data_rep, parameters)
    return prediction

# getPID
def getPID(text: list, im, rotation=0):
    def containsPIDVar(text: list):
        for entry in text:
            if "PID" in entry:
                return True, entry
        return False, ""

    truth, name = containsPIDVar(text)
    if truth:
        return True, text[text.index(name) + 1]
    elif rotation != 3:
        received, answer = getPID(pytesseract.image_to_data(np.rot90(im), output_type=Output.DICT)['text'], np.rot90(im), rotation + 1)
        return received, answer
    else:
        return False, ""

# getSER
def getSER(text: list, im, rotation=0):
    def containsSERVar(text: list):
        for entry in text:
            if "S/N" in entry:
                return True, entry
            elif "Ser" in entry:
                return True, entry
        return False, ""
    truth, name = containsSERVar(text)
    if truth:
        return True, text[text.index(name) + 1]
    else:
        return False, ""

# getMOD
def getMOD(image, drive_types: list, text: list, training_data_loc: str):
    def containsMODVar(text:list):
        for entry in text:
            if "MODEL" in entry.upper():
                return True, entry
        return False, ""
    
    def processMODEntry(mod_num: str):
        special_chars = {"*", "\"", " ", "+", ":", "\n"}
        for special_char in special_chars:
            while special_char in mod_num:
                if mod_num.index(special_char) == 0:
                    mod_num = mod_num[1:]
                elif mod_num.index(special_char) == len(mod_num) - 1:
                    mod_num = mod_num[:-1]
                else:
                    mod_num = mod_num[0:mod_num.index(special_char)] + mod_num[mod_num.index(special_char) + 1:]
        return mod_num
    
    model_params = load_models(training_data_loc)[0]
    model_drive = predict(image, model_params[0], BWHistogram(), model_params[1])
    for drive_type in drive_types:
        if drive_type in model_drive:
            model_drive = model_drive[0:model_drive.index(drive_type)]
            break

    truth, name = containsMODVar(text)
    if truth:
        app_ind = 1
        while len(text[text.index(name) + app_ind]) < 10 and text[text.index(name) + app_ind] not in model_drive:
            app_ind += 1
        if text[text.index(name) + app_ind] in model_drive:
            model_success = 1
            return model_drive, model_success, model_drive
        elif processMODEntry(text[text.index(name) + app_ind]) == model_drive:
            model_success = 1
        else:
            model_success = 0
        return processMODEntry(text[text.index(name) + app_ind]), model_success, model_drive
    return model_drive, -1, ""

"""
drives = ["DELL", "HP", "SEAGATE", "HP", "SAMSUNG", "HGST", "LENOVO"]
print("RECEIVED PID: {0}".format(getPID(scanned_text['text'])[1]))

model_no, model_success_int, model_prediction = getMOD(drives, scanned_text['text'])
print("RECEIVED MODEL NUMBER: {0}".format(model_no))

if model_success_int == 0:
    model_success = "FAILED"
elif model_success_int == 1:
    model_success = "SUCCEEDED"
else:
    model_success = "UNKNOWN"

if model_success_int == -1:
    print("MODEL SUCCESS: {0}".format(model_success))
else:
    print("MODEL SUCCESS: {0}\nPREDICTED: {1}".format(model_success, model_prediction))
"""