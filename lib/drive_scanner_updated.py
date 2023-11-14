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
from models_updated import CModel, KNearest, ModelUtils
from preprocessing import *
from data_representation_abstracted_updated import TrainingRepresentation, BWHistogram
from bow import BOW

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
def load_models(trained_data_loc, K=1, accuracy_mode=False):
    model_params = []

    KN_BW_X_loc = trained_data_loc + "\\KNearest_BWHist_X.hdf5"
    KN_BW_Y_loc = trained_data_loc + "\\KNearest_BWHist_Y.hdf5"
    KN_BW_Dict_loc = trained_data_loc + "\\KNearest_BWHist_Dict.yaml"

    k_model = KNearest()
    k_model.load(KN_BW_X_loc, KN_BW_Y_loc, KN_BW_Dict_loc)
    parameters = {'K': K, 'optimal_size': len(k_model._KNearest__training[0][1]), 'accuracy_mode': accuracy_mode}
    model_params.append([k_model, parameters.copy()])


    KN_BOW_X_loc = trained_data_loc + "\\KNearest_BOW_X.hdf5"
    KN_BOW_Y_loc = trained_data_loc + "\\KNearest_BOW_Y.hdf5"
    KN_BOW_Dict_loc = trained_data_loc + "\\KNearest_BOW_Dict.yaml"
    b_model = KNearest()
    b_model.load(KN_BOW_X_loc, KN_BOW_Y_loc, KN_BOW_Dict_loc)
    parameters = {'K': K, 'accuracy_mode': accuracy_mode}

    model_params.append([b_model, parameters.copy()])

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
def getMOD(image, drive_types: list, text: list, training_data_locs: list, accuracy_mode=False):
    def containsMODVar(text:list):
        for entry in text:
            if "MODEL" in entry.upper():
                return True, entry
        return False, ""

    trained_models = load_models(training_data_locs, 3, accuracy_mode)
    BW_model = trained_models[0]
    BOW_model = trained_models[1]

    model_predictions = {}

    model_drive = predict(image, BW_model[0], BWHistogram(), BW_model[1])
    model_drive_2 = predict(image, BOW_model[0], BOW("BOW.txt"), BOW_model[1])
    #These are returning lists of potential matches

    model_predictions["BWHist"] = model_drive
    model_predictions["BOW"] = model_drive_2

    Y_dict = {}
    with open(training_data_locs + "\\" + "KNearest_BWHist_Dict.yaml") as yamlfile:
        Y_dict = yaml.safe_load(yamlfile)

    truth, name = containsMODVar(text)

    if truth:
        app_ind = 1
        while len(text[text.index(name) + app_ind]) < 10 and text.index(name) + app_ind < len(text):
            app_ind += 1
        if text.index(name) + app_ind < len(text) - 1 and text[text.index(name) + app_ind]:
            model_predictions["OCR"] = text[text.index(name) + app_ind]

    return ModelUtils.get_overall_prediction(model_predictions, drive_types, list(Y_dict.keys()))