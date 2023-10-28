from data_representation_abstracted_copy1 import *
import numpy as np
import yaml
import cv2

class CModel():
    def predict(image, data_rep_alg: TrainingRepresentation, parameters: dict):
        pass
    def train():
        pass

class DistanceMetrics():
    def manhattan_dist(self, x, y):
        return np.sum(np.abs(x - y))
    def euclid_dist(self, x, y):
        return (np.sum((x - y)**2))**(1/2)

class KNearest(CModel):
    
    __training: list

    def predict(self, image, data_rep_alg: TrainingRepresentation, parameters: dict):
        if self.__training is None or len(self.__training) == 0:
            print("ERROR. NO TRAINING DATA. PLEASE CALL train() FIRST.")
            exit(1)

        data_rep = data_rep_alg.represent_data(image, parameters)
        
        labels = {}
        for rep in self.__training:
            labels[rep[0]] = 0

        #run K-nearest
        distances = []
        categories = []
        for rep in self.__training:
            categories.append(rep[0])
            distances.append(DistanceMetrics().manhattan_dist(np.array(data_rep), np.array(rep[1])))
        
        categories = np.array(categories)
        distances = np.array(distances)

        for _ in range(parameters['K']):
            labels[categories[np.argmin(distances)]] += 1
        
        frequencies = list(labels.values())
        cats = list(labels.keys())

        return cats[frequencies.index(max(frequencies))]


    def train(self, file_loc: str):
        with open(file_loc, "r") as file:
            self.__training = yaml.safe_load(file)


img_labels = []
with open("data\\15021026 1 fixed.csv") as csv_file:
    table_reader = csv.DictReader(csv_file)
    for row in table_reader:
        img_labels.append([row['PID'], row['Model'] + row['Manufacturer']])

PIDS = [label[0] for label in img_labels]
labels = [label[1] for label in img_labels]

images = ["data\\" + str(PID) + ".jpg" for PID in PIDS] 
kmodel = KNearest()
kmodel.train("trained_means.yaml")

num_correct = 0
total_num = 0

dict_accuracy = {}

with open("trained_means_results_f.txt", "w") as res_file:
    for i, image in enumerate(images):
        result = kmodel.predict(cv2.imread(image), BWHistogram(), {"optimal_size": 3024, 'K': 1})
        if labels[i] == result:
            num_correct += 1
            if labels[i] in list(dict_accuracy.keys()):
                dict_accuracy[labels[i]][0] += 1
        total_num += 1
        if labels[i] not in list(dict_accuracy.keys()):
            dict_accuracy[labels[i]] = [0, 1]
        else:
            dict_accuracy[labels[i]][1] += 1
        print(res_file, "Prediction Accuracy: {0} at label {1}".format(num_correct/total_num, labels[i]))
    for key in list(dict_accuracy.keys()):
        print(res_file, "Accuracy for {0} Label: {1} | Number Considered: {2}".format(key, dict_accuracy[key][0], dict_accuracy[key][1]))


"""
img_labels = []
with open("data\\15021026 1.csv") as csv_file:
    table_reader = csv.DictReader(csv_file)
    for row in table_reader:
        img_labels.append([row['PID'], row['Model'] + row['Manufacturer']])

PIDS = [label[0] for label in img_labels]
labels = [label[1] for label in img_labels]

images = ["data\\" + str(PID) + ".jpg" for PID in PIDS] 
kmodel = KNearest()
kmodel.train("trained_means.yaml")

num_correct = 0
total_num = 0

dict_accuracy = {}
for i, image in enumerate(images):
    result = kmodel.predict(cv2.imread(image), BWHistogram(), {"optimal_size": 3024, 'K': 1})
    if labels[i] == result:
        num_correct += 1
        if labels[i] in list(dict_accuracy.keys()):
            dict_accuracy[labels[i]][0] += 1
    total_num += 1
    if labels[i] not in list(dict_accuracy.keys()):
        dict_accuracy[labels[i]] = [0, 1]
    else:
        dict_accuracy[labels[i]][1] += 1
    print("Prediction Accuracy: {0} at label {1}".format(num_correct/total_num, labels[i]))
for key in list(dict_accuracy.keys()):
    print("Accuracy for {0} Label: {1} | Number Considered: {2}".format(key, dict_accuracy[key][0], dict_accuracy[key][1]))
"""