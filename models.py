from data_representation_abstracted_copy1 import *
import numpy as np
import yaml

class CModel():
    def predict(image, data_rep_alg: TrainingRepresentation, parameters: dict):
        pass
    def train():
        pass

class DistanceMetrics():
    def manhattan_dist(x, y):
        return np.sum(np.abs(x - y))
    def euclid_dist(x, y):
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
            distances.append(DistanceMetrics().manhattan_dist(data_rep, rep[1]))
        
        categories = np.array(categories)
        distances = np.array(distances)

        best_points = []
        for _ in parameters['K']:
            labels[categories[np.argmax(distances)]] += 1
        
        frequencies = list(labels.values())
        cats = list(labels.keys())

        return cats[frequencies.index(max(frequencies))]


    def train(self, file_loc: str):
        self.__training = yaml.safe_load(file_loc)