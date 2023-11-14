from data_representation_abstracted_updated import TrainingRepresentation
from utilities import convert_all_to_one_hots

import numpy as np
import yaml
import cv2
import h5py

class CModel():
    def predict(image, data_rep_alg: TrainingRepresentation, parameters: dict):
        pass

    @staticmethod
    def load(X_loc, Y_loc, dict_loc):
        f1 = h5py.File(X_loc, "r")
        f2 = h5py.File(Y_loc, "r")

        X = np.array(f1["data"])
        Y = np.array(f2["labels"])
        Y_dict = {}

        with open(dict_loc, "r") as ymlfile:
            Y_dict = yaml.safe_load(ymlfile)
        
        return X, Y, Y_dict

    def train(self, training_rep: TrainingRepresentation, image_folder: str, csv_path: str, num_threads=10):
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

        ### REPRESENTS EACH IMAGE ACCORDING TO PASSED TRAINING-REPRESENTER ###
        def represent_training(training_labels: dict, training_rep: TrainingRepresentation, parameters={}):
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
                    results = training_rep.represent_data(data['image'], parameters)
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
        ### END HELPER METHODS ###

        training_data = process_training(load_training(image_folder, csv_path))
        represented_training = represent_training(training_data, training_rep, parameters={'optimal_size': training_data[list(training_data.keys())[0]]['image'].shape[1]})
        Y = [entry[0] for entry in represented_training]
        X = [entry[1] for entry in represented_training]
        return X, convert_all_to_one_hots(Y)

class ModelUtils():
    MODEL_WEIGHTS = {"BWHist": 100, "BOW": 50}
    ERROR_THRESHOLD = 2

    @staticmethod
    def get_overall_prediction(model_dict: dict, drive_types: list, labels: list):
        def strcomp(s1, s2):
            error = np.abs(len(s2) - len(s1))
            for i in range(len(s1)):
                if i == len(s2):
                    break
                if s2[i] != s1[i]:
                    error += 1
            return error
        
        def takeout_brand(s1: str, brands: list):
            for brand in brands:
                if brand in s1:
                    return s1[0:s1.index(brand)]
            return s1
        
        results = {}
        for label in labels:
            results[label] = 0

        labels_without_brands = [takeout_brand(label, drive_types) for label in labels]
        
        ocrFlag = False
        if "OCR" in list(model_dict.keys()):
            if model_dict["OCR"].strip() in labels_without_brands: #If the OCR picked up the model number, use it
                return model_dict["OCR"].strip()
            else:
                ocrFlag = True
        
        #Want to decide between BWHist and BOW
        BWHist = model_dict["BWHist"]
        BOW = model_dict["BOW"]

        for label in BWHist:
            results[label] += ModelUtils.MODEL_WEIGHTS["BWHist"]
        for label in BOW:
            results[label] += ModelUtils.MODEL_WEIGHTS["BOW"]
        
        list_build = {}
        if ocrFlag:
            for prediction in BWHist:
                list_build[prediction] = strcomp(takeout_brand(prediction, drive_types), model_dict["OCR"].strip())
            for prediction in BOW:
                list_build[prediction] = strcomp(takeout_brand(prediction, drive_types), model_dict["BOW"].strip())
            
            for prediction in list(list_build.keys()):
                results[prediction] -= list_build[prediction]
        
        result_labels = list(results.keys())
        scores = list(results.values())

        return result_labels[scores.index(max(scores))]
    
class DistanceMetrics():
    def manhattan_dist(self, x, y):
        return np.sum(np.abs(x - y))
    def euclid_dist(self, x, y):
        return (np.sum((x - y)**2))**(1/2)

class KNearest(CModel):
    
    __training: list

    def predict(self, image, data_rep_alg: TrainingRepresentation, parameters: dict):
        if self.__training is None or len(self.__training) == 0:
            print("ERROR. NO TRAINING DATA. PLEASE CALL load() FIRST.")
            exit(1)

        #Load in training set
        X = self.__training[0]
        Y = self.__training[1]
        Y_dict = {tuple(value): key for key, value in self.__training[2].items()}

        data_rep = data_rep_alg.represent_data(image, parameters)
        
        #Basically takes all of the possible labels and creates a dictionary with all zeroes for prediction purposes
        labels = {}
        for rep in set([tuple([int(x) for x in entry]) for entry in list(self.__training[1])]):
            labels[rep] = 0

        #run K-nearest
        distances = []
        for x in X:
            if 'accuracy_mode' in list(parameters.keys()):
                if parameters['accuracy_mode']:
                    distance = DistanceMetrics().manhattan_dist(np.array(data_rep), np.array(x))
                    if distance == 0:
                        distance = 1000000000000
                    distances.append(distance)
                else:
                    distances.append(DistanceMetrics().manhattan_dist(np.array(data_rep), np.array(x)))
            else:
                distances.append(DistanceMetrics().manhattan_dist(np.array(data_rep), np.array(x)))
        
        distances = np.array(distances)
        sorted_copy = np.sort(distances)

        K_nearest_points = []
        i = 0
        while i < parameters['K']:
            n = 0
            possibilities = np.where(distances==sorted_copy[i])[0]
            for k in possibilities:
                if i + n == range(parameters['K']):
                    break
                K_nearest_points.append([sorted_copy[i], Y[k]])
                n += 1
            i += n

        
        for i in range(len(K_nearest_points)):
            K_nearest_points[i][0] = K_nearest_points[-1][0] - K_nearest_points[i][0]
        
        for i in range(len(K_nearest_points)):
            labels[tuple([int(x) for x in list(K_nearest_points[i][1].flatten())])] += K_nearest_points[i][0]

        scores = list(labels.values())
        cats = list(labels.keys())

        sorted_copy = list(np.sort(scores))[::-1]
        top_score_length = 0
        for x in sorted_copy:
            if x == sorted_copy[0]:
                top_score_length += 1
            else:
                break

        if top_score_length > len(sorted_copy):
            top_score_length = len(sorted_copy)
        best_scores = [Y_dict[cats[scores.index(sorted_copy[i])]] for i in range(top_score_length)]
        return list(set(best_scores))

    def train(self, training_rep: TrainingRepresentation, image_folder: str, csv_path: str, num_threads=10):
        return super().train(training_rep, image_folder, csv_path, num_threads)
    
    def load(self, X_loc, Y_loc, Y_dict_loc):
        X, Y, Y_dict = super().load(X_loc, Y_loc, Y_dict_loc)
        self.__training = [X, Y, Y_dict]