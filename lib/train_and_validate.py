import subprocess
import os

from random import Random

import h5py
import numpy as np

import argparse

def get_inverse_exclusions(all_relevant_files: list, exclusions: list):
    return [file[len(image_arg + "\\"):file.find(".jpg")] for file in all_relevant_files if file[len(image_arg + "\\"):file.find(".jpg")] not in exclusions]

image_arg = "..\\..\\\\data"
model_folder = "training_and_validation"
batch = "True"
test_result_file = "test_results.txt"
training_result_file = "training_results.txt"

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("images",
                help="path to images to scan")
ap.add_argument("csv",
                help="path to formatted csv file")
ap.add_argument("-e", "--extension",
                help="file extension to load in",
                default=".jpg")
args = vars(ap.parse_args())

image_arg = args["images"]
csv_path = args["csv"]

TEST_SET_LENGTH = 78

all_files = os.listdir(image_arg)
selected_files = [os.path.join(image_arg, file) for file in all_files if file.endswith(".jpg")]

ratio = int((TEST_SET_LENGTH / len(selected_files)) * 100)

test_set = []
const_add = False
for i, file in enumerate(selected_files):
    rand_select = Random().randint(1, 100)
    if const_add or (rand_select < ratio and len(test_set) < TEST_SET_LENGTH):
        test_set.append(file[len(image_arg + "\\"):file.find(".jpg")])
    if len(test_set) >= TEST_SET_LENGTH:
        break
    if len(selected_files) - i == TEST_SET_LENGTH - len(test_set):
        const_add = True

exclusion_list = ""
for PID in test_set:
    exclusion_list += PID + ","
exclusion_list = exclusion_list.strip()[:-1]

inv_exclusions_list = ""
inv_exclusion_list_l = get_inverse_exclusions(selected_files, exclusion_list.split(','))
for name in inv_exclusion_list_l:
    inv_exclusions_list += name + ","
inv_exclusions_list = inv_exclusions_list.strip()[:-1]

result = subprocess.run(["python", "lib\\drive_scanner_trainer.py", "-i", image_arg, "-c", csv_path, "-t", model_folder, "--exclusions", exclusion_list])
#got results in a text file

#will exclude all but test images (runs on test images only)
test_result = subprocess.run(["python", "lib\\drive_scanner_runner.py", "-i", image_arg, "-b", "True", "-t", model_folder, "--exclusions", inv_exclusions_list, "-r", test_result_file])
print("TEST COMPLETE")

#will exclude all but training images (runs on training images only)
training_result = subprocess.run(["python", "lib\\drive_scanner_runner.py", "-i", image_arg, "-b", "True", "-t", model_folder, "--exclusions", exclusion_list, "-r", training_result_file])
print("TRAINING COMPLETE")

