import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocess
import cv2
from skimage.filters import threshold_otsu
import csv

from queue import *
from threading import Thread

DELL_type1_images = [cv2.imread("data\\4421175.jpg"), cv2.imread("data\\4421195.jpg"), cv2.imread("data\\4421177.jpg"), cv2.imread("data\\4421183.jpg")]
DELL_type1_images_codes = [[],[],[],[]]

SEAGATE_images = [cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421647.jpg"), cv2.imread("data\\4421649.jpg")]
SEAGATE_images_codes = [[],[],[],[]]

#dictionary?
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
            r_queue.put((data['label'], results))
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

        return np.array(completed)


        
    feature_vector = []
    # 1) preprocess
    b_w_image = Preprocess.gaussian_blur(Preprocess.to_bw(Preprocess.remove_bg(image)), 1)
    processed_image = Preprocess.to_binary(b_w_image, threshold_otsu(b_w_image))
    # 2) convert to histogram array
    sums = np.sum(processed_image, axis=0)
    reg_sum = sums / max(sums)
    feature_vector = regularize(remove_white(reg_sum))

    # 3) smooth?    
    return feature_vector

training_data = process_training(load_training("data", "data\\15021026 1.csv"))
import pdb; pdb.set_trace()
represented_training = represent_training(training_data, parameters={'optimal_size': training_data[list(training_data.keys())[0]]['image'].shape[1]})
pass

"""
threshold = 180
for i in range(len(DELL_type1_images)):
    DELL_type1_images[i] = 255 - cv2.cvtColor(remove(DELL_type1_images[i]), cv2.COLOR_RGB2GRAY)
    DELL_type1_images[i] = np.where(DELL_type1_images[i] < threshold, 0, 1)
    for k in range(DELL_type1_images[i].shape[1]):
        DELL_type1_images_codes[i].append(np.sum(DELL_type1_images[i][:, k]))
    DELL_type1_images_codes[i] /= np.max(DELL_type1_images_codes[i])



for i in range(len(SEAGATE_images)):
    SEAGATE_images[i] = 255 - cv2.cvtColor(remove(SEAGATE_images[i]), cv2.COLOR_RGB2GRAY)
    SEAGATE_images[i] = np.where(SEAGATE_images[i] < threshold, 0, 1)
    for k in range(SEAGATE_images[i].shape[1]):
        SEAGATE_images_codes[i].append(np.sum(SEAGATE_images[i][:, k]))
    SEAGATE_images_codes[i] /= np.max(SEAGATE_images_codes[i])

images = [DELL_type1_images, SEAGATE_images]
image_codes = [DELL_type1_images_codes, SEAGATE_images_codes]
rows = len(images)

columns = len(image_codes[0])
fig, ax = plt.subplots(rows, columns, sharex='col', sharey='row')

for row in range(rows):
    for col in range(columns):
        btm_labels = np.arange(len(image_codes[row][col]))
        ax[row,col].bar(btm_labels, image_codes[row][col])

plt.show()

avg_curves = []
for code_list in range(len(image_codes)):
    avg_curves.append(np.sum(image_codes[code_list], axis=0)/len(image_codes[code_list][0]))

rows = len(images)
fig, ax = plt.subplots(rows, 1, sharex='col', sharey='row')
for row in range(rows):
    btm_labels = np.arange(len(avg_curves[row]))
    ax[row].bar(btm_labels, avg_curves[row])

plt.show()

pass
"""