from models_updated import *
from data_representation_abstracted_updated import *
from bow import BOW
from utilities import ConstantNames
import h5py
import yaml

import argparse
import os

def load_to_file(trained_model_parameters, trained_data_folder, data_rep_name, ml_alg_name):
    saved_model = h5py.File(trained_data_folder + "\\" + "model" + ".hdf5", "a")

    entries = list(trained_model_parameters.keys())
    for entry in entries:
        if entry != "Y_dict":
            saved_model.create_dataset(entry + "_" + data_rep_name + "_" + ml_alg_name, data=trained_model_parameters[entry])

    file_name_dict = trained_data_folder + "\\" + data_rep_name + "_" + ml_alg_name + "_Dict"

    keys = list(trained_model_parameters['Y_dict'].keys())
    conv_dict = {}
    for key in keys:
        conv_dict[key] = [float(element) for element in list(trained_model_parameters['Y_dict'][key])]

    with open(file_name_dict + ".yaml", "w") as ymlfile:
        yaml.safe_dump(conv_dict, ymlfile)
    saved_model.close()
    pass

def train_on(ml_alg: CModel, data_rep: TrainingRepresentation, ml_alg_name: str, data_rep_name: str):
    X, Y_arr = CModel.represent_images_as_data(data_rep, "..\\data", "..\\data\\15021026 1 fixed.csv")
    Y_dict = Y_arr[0]
    Y = Y_arr[1]

    trained_model_parameters = ml_alg.train(X, Y, Y_dict)

    return load_to_file(trained_model_parameters, args['trained_data'], data_rep_name, ml_alg_name)

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

#Setting up training
train_on(KNearest(), BWHistogram(), ConstantNames.KNEAREST, ConstantNames.BWHIST)
saved_model = train_on(KNearest(), BOW("BOW.txt"), ConstantNames.KNEAREST, ConstantNames.BOW)
saved_model.close()

