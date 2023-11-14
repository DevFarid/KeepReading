from models_updated import *
from data_representation_abstracted_updated import *
from bow import BOW
import h5py
import yaml

import argparse
import os

def load_to_file(file_name_X: str, file_name_Y: str, file_name_dict: str, X, Y, Y_dict):
    fX = h5py.File(file_name_X + ".hdf5", "w")
    fY = h5py.File(file_name_Y + ".hdf5", "w")

    fX.create_dataset("data", data=X)
    fY.create_dataset("labels", data=Y)

    fX.close()
    fY.close()

    keys = list(Y_dict.keys())
    conv_dict = {}
    for key in keys:
        conv_dict[key] = [float(element) for element in list(Y_dict[key])]

    with open(file_name_dict + ".yaml", "w") as ymlfile:
        yaml.safe_dump(conv_dict, ymlfile)
    pass

def train_on(ml_alg: CModel, data_rep: TrainingRepresentation, ml_alg_name: str, data_rep_name: str):
    X, Y_arr = ml_alg.train(data_rep, "..\\data", "..\\data\\15021026 1 fixed.csv")
    Y_dict = Y_arr[0]
    Y = Y_arr[1]

    load_to_file(args['trained_data'] + "\\" + ml_alg_name + "_" + data_rep_name + "_X", args['trained_data'] + "\\" + ml_alg_name + "_" + data_rep_name + "_Y", args['trained_data'] + "\\" + ml_alg_name + "_" + data_rep_name + "_Dict", X, Y, Y_dict)

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",
                required=True,
                help="path to images to scan")
ap.add_argument("-e", "--extension",
                help="file extension to load in",
                default=".jpg")
ap.add_argument("-t", "--trained_data",
                help="folder name of location where trained data should be saved",
                default="model")
args = vars(ap.parse_args())

if not os.path.exists(args['trained_data']):
    os.mkdir(args['trained_data'])

"""
#Training BWHistogram Representation with KNearest
X, Y_arr = KNearest().train(BWHistogram(), "..\\data", "..\\data\\15021026 1 fixed.csv")
Y_dict = Y_arr[0]
Y = Y_arr[1]

load_to_file(args['trained_data'] + "\\KNearest_BWHist_X", args['trained_data'] + "\\KNearest_BWHist_Y", args['trained_data'] + "\\KNearest_BWHist_Dict", X, Y, Y_dict)

#Training BOW Representation with KNearest
X, Y_arr = KNearest().train(BOW("BOW.txt"), "..\\data", "..\\data\\15021026 1 fixed.csv")
Y_dict = Y_arr[0]
Y = Y_arr[1]

load_to_file(args['trained_data'] + "")
"""

train_on(KNearest(), BWHistogram(), "KNearest", "BWHist")
train_on(KNearest(), BOW("BOW.txt"), "KNearest", "BOW")

