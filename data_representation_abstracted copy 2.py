import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocess
import cv2
from skimage.filters import threshold_otsu
import csv
import yaml

from queue import *
from threading import Thread

def load_training(image_folder, csv_path) -> dict:
    drive_dict = {}
    with(open(csv_path)) as csv_file:
        tablereader = csv.DictReader(csv_file)
        for row in tablereader:
            row['Image'] = cv2.imread(image_folder + "\\" + row['PID'] + ".jpg")
            drive_dict[row['PID']] = row
    return drive_dict

def process_training(drive_dict: dict):
    training_labels = {}
    PIDs = list(drive_dict.keys())
    for PID in PIDs:
        training_labels[PID] = {'label': drive_dict[PID]['Model'] + drive_dict[PID]['Manufacturer'], 'image': drive_dict[PID]['Image']}
    return training_labels

num_threads = 10
def represent_training(training_labels: dict, parameters: dict):
    represented_training = []
    PIDs = list(training_labels.keys())

    d_queue = Queue()
    r_queue = Queue()

    def worker():
        while True:
            thing = d_queue.get()
            if thing is None:
                break
            data = thing[1]
            results = represent_data(data['image'], parameters)
            r_queue.put([data['label'], results])
            d_queue.task_done()

            print("representing {0}...".format(thing[0]))


    for PID in PIDs:
        d_queue.put((PID, training_labels[PID]))
    for n_thread in range(num_threads):
        d_queue.put(None)

    threads = []
    for _ in range(num_threads):
        thread = Thread(target=worker)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    while not r_queue.empty():
        represented_training.append(r_queue.get())

    return represented_training

def represent_data(image, parameters: dict = {}):
    def remove_white(arr):
        removed_whitespace = []
        for val in arr:
            if val > 0:
                removed_whitespace.append(val)
        return np.array(removed_whitespace)

    def regularize(arr):
        means = []
        completed = list(arr).copy()
        for i in range(len(arr) - 1):
            means.append((arr[i] + arr[i + 1]) / 2)
        
        i = 0
        index = 1
        while len(completed) < parameters['optimal_size']:
            if i < len(means):
                completed.insert(index, means[i])
            else:
                break
            index += 2
            i += 1

        while len(completed) > parameters['optimal_size']:
            completed.pop()

        return completed


        
    feature_vector = []
    # 1) preprocess
    b_w_image = Preprocess.gaussian_blur(Preprocess.to_bw(Preprocess.remove_bg(image)), 1)
    processed_image = Preprocess.to_binary(b_w_image, threshold_otsu(b_w_image))
    # 2) convert to histogram array
    sums = np.sum(processed_image, axis=0)
    reg_sum = sums / max(sums)
    feature_vector_p = regularize(remove_white(reg_sum))

    # 3) smooth?
    for num in feature_vector_p:
        feature_vector.append(float(num))
    return feature_vector


training_data = process_training(load_training("data", "data\\15021026 1.csv"))
represented_training = represent_training(training_data, parameters={'optimal_size': training_data[list(training_data.keys())[0]]['image'].shape[1]})
with open("trained_reps.yaml", "w") as file:
    yaml_file = yaml.safe_dump(represented_training, file)

print("SUCCESS!")