import cv2
from bow_tester import *
from OCR import ObjectCharacterRecognition
from concurrent.futures import ThreadPoolExecutor

from data_representation_abstracted_updated import *
from utilities import ConstantFilePaths

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
        ocrResults = ObjectCharacterRecognition.getResults(img)
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
        return list(BOW(ConstantFilePaths().bow).search(im).values())
          
    def getDictionary(self) -> dict:
        return self.bag_of_words

    