import cv2
from bow_tester import *
from OCR import *
from concurrent.futures import ThreadPoolExecutor

from data_representation_abstracted_updated import *

class BOW(TrainingRepresentation):
    """
        Bag of Words
        Created by Farid Kamizi on 10/19/2023.
        This class represents an algorithm called `Bag of Words`.
        Given a "document", our goal is to search for interested words in that document.
    """     
    def __init__(self, file) -> None:
        self.bag_of_words = list()
        self.loadWordsFromFile(file)
        print("BOW Vector = ", self.bag_of_words)

    # TODO: load words from file.
    def loadWordsFromFile(self, file) -> None:
        with open(file, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                if len(line) != 0:
                    if line.find("\n") == 0 or line.find("//") == -1:
                        self.bag_of_words.append(line)

    def setDictionary(self, newDict) -> None:
        self.bag_of_words = newDict

    def addWord(self, word) -> None:
        self.bag_of_words.append(word)

    def removeWord(self, word) -> None:
        return self.bag_of_words.remove(word) if self.bag_of_words.__contains__(word) else None
    
    # runs bag of words on OCR results.
    def search(self, img) -> dict:
        bow_vector = [0] * len(self.bag_of_words)
        ocrResults = OCR.getResults(img)
        for word in self.bag_of_words:
            for result in ocrResults["text"]:
                if result.find(word) != -1:
                    bow_vector[self.bag_of_words.index(word)] = 1
        return {k: v for k, v in zip(self.bag_of_words, bow_vector)}


    def test(self, label, results, min_conf=0) -> dict:
        bow_representation = self.search(results, min_conf)
        print("Label: " + str(label))
        print(bow_representation)
        print("")
        return bow_representation
    
    def represent_data(self, im, param: dict):
        return list(BOW("Model\\BOW.txt").search(im).values())
          
    def getDictionary(self) -> dict:
        return self.bag_of_words
    

def represent_training2(training_labels: dict, parameters={}):
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
            results = BOW.represent_data(data['image'], parameters)
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
    
if __name__ == "__main__":
    training_data = process_training(load_training("data", "data\\15021026 1 fixed.csv"))
    represented_training = represent_training2(training_data)

    with open("average_bow.yaml", "w") as yml_file:
        yaml.safe_dump(compute_averages(represented_training), yml_file)
"""
    bow = BOW('bow/BOW.txt')
    x = list(bow.search(cv2.imread("4421225.jpg")).values())
    print(x)
"""
    