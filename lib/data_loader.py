import cv2
import csv
import numpy as np

from threading import Thread
from queue import Queue
from bow import BOW

### LOADS EVERY IMAGE IN ###
def load_training(image_folder, csv_path) -> dict:
    drive_dict = {}
    with(open(csv_path)) as csv_file:
        tablereader = csv.DictReader(csv_file)
        for row in tablereader:
            row['Image'] = cv2.imread(image_folder + "\\" + row['PID'] + ".jpg")
            drive_dict[row['PID']] = row
    return drive_dict

### LABELS EACH IMAGE ###
def process_training(drive_dict: dict):
    training_labels = {}
    PIDs = list(drive_dict.keys())
    for PID in PIDs:
        training_labels[PID] = {'label': drive_dict[PID]['Model'] + drive_dict[PID]['Manufacturer'], 'image': drive_dict[PID]['Image']}
    return training_labels

### REPRESENTS EACH IMAGE ACCORDING TO BOW RIGHT NOW BUT SHOULD DO PASSED IN TRAINING-REPRESENTER ###
num_threads = 10
def represent_training(training_labels: dict, parameters={}):
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
            results = BOW().represent_data(data['image'], parameters)
            r_queue.put([data['label'], results])
            d_queue.task_done()

            print("representing {0}...".format(thing[0]))


    for PID in PIDs:
        d_queue.put((PID, training_labels[PID]))
    for _ in range(num_threads):
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

### COMPUTES THE AVERAGES OF THE SUPPLIED REPRESENTATIONS ###
def compute_averages(img_data: list):
    avg_dict = {}
    count_dict = {}
    result = []
    for entry in img_data:
        if entry[0] not in list(avg_dict.keys()):
            avg_dict[entry[0]] = np.array(entry[1])
            count_dict[entry[0]] = 1
        else:
            avg_dict[entry[0]] += np.array(entry[1])
            count_dict[entry[0]] += 1
    temp_result = []
    for key in list(avg_dict.keys()):
        temp_result.append([key, avg_dict[key] / count_dict[key]])
    for entry in temp_result:
        fixed_list = []
        for num in entry[1]:
            fixed_list.append(float(num))
        result.append([entry[0], fixed_list])
    return result

### CONVERTS A CATEGORY TO A ONE-HOT ###
def convert_one_hot(label: str, label_dict: dict):
    if label in list(label_dict.keys()):
        return label_dict[label], label_dict
    elif len(list(label_dict.keys())) == 0:
        label_dict[label] = [1]
        return [1], label_dict
    else:
        fixed_dict = {}
        opt_length = 0
        for key in list(label_dict.keys()):
            fixed_dict[key] = np.append(label_dict[key], [0])
            opt_length = len(fixed_dict[key])
        fixed_dict[label] = np.append(np.zeros((opt_length - 1,)), [1])
        return fixed_dict[label], fixed_dict

### CONVERTS ALL CATEGORIES TO ONE HOTS ###
def convert_all_to_one_hots(labels):
    one_hot_dict = {}

    for label in labels:
        _, one_hot_dict = convert_one_hot(label, one_hot_dict)

    return one_hot_dict, np.array([one_hot_dict[l] for l in labels])

"""
training_data = process_training(load_training("data", "data\\15021026 1.csv"))
represented_training = represent_training(training_data, parameters={'optimal_size': training_data[list(training_data.keys())[0]]['image'].shape[1]})

Y = []
X = []
for piece_of_data in represented_training:
    Y.append(piece_of_data[0])
    X.append(piece_of_data[1])
convert_dict, Y = convert_all_to_one_hots(np.array(Y))
X = np.array(X)

print("HERE")
file = h5py.File('data_labels.hdf5', 'w')
dset = file.create_dataset('data_labels', data=Y)
file.close()

file = h5py.File('data.hdf5', 'w')
dset = file.create_dataset('dataset', data=X)
file.close()
"""

"""
with open("trained_means.yaml", "w") as file:
    yaml_file = yaml.safe_dump(compute_averages(represented_training), file)
"""

"""
with open("trained_reps.yaml", "w") as file:
    yaml_file = yaml.safe_dump(represented_training, file)
"""