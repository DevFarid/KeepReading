import cv2
from bow_tester import *
from OCR import *
from concurrent.futures import ThreadPoolExecutor

class BOW:
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
    def search(self, img, min_conf=0) -> dict:
        bow_vector = [0] * len(self.bag_of_words)
        ocrResults = OCR.getResults(img)
        for word in self.bag_of_words:
            for result in ocrResults["text"]:
                if result.find(word) != -1:
                    bow_vector[self.bag_of_words.index(word)] = 1
        # return {k: v for k, v in zip(self.bag_of_words, bow_vector)}
        return bow_vector


    def reduce(self, R: list, N: int) -> list:
        result = []
        i = 0
        while i < len(R):
            sum = 0
            for j in range(N):
                sum += R[i+j]
            result.append(sum)
            i += N
        return result

    def test(self, label, results, min_conf=0) -> dict:
        bow_representation = self.search(results, min_conf)
        print("Label: " + str(label))
        print(bow_representation)
        print("")
        return bow_representation
          
    def getDictionary(self) -> dict:
        return self.bag_of_words
    
if __name__ == "__main__":
    bow = BOW('bow/BOW.txt')
    x = bow.search(cv2.imread("data/4421310.jpg"))
    print(f"Resulting vector: {x}\nDimensionality Reduced: {bow.reduce(x, 2)}")
    