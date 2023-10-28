import cv2
from bow_tester import *

class BOW:
    """
        Bag of Words
        Created by Farid Kamizi on 10/19/2023.
        This class represents an algorithm called `Bag of Words`.
        Given a "document", our goal is to search for interested words in that document.
    """     
    def __init__(self, file) -> None:
        self.words = list()
        self.loadWordsFromFile(file)
        print("BOW Vector = ", self.words)

    # TODO: load words from file.
    def loadWordsFromFile(self, file) -> None:
        with open(file, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                if line.find("\n") == 0 or line.find("//") == -1:
                    self.words.append(line)

    def setDictionary(self, newDict) -> None:
        self.words = newDict

    def addWord(self, word) -> None:
        self.words.append(word)

    def removeWord(self, word) -> None:
        return self.words.remove(word) if self.words.__contains__(word) else None
    
    # runs bag of words on OCR results.
    def search(self, img, min_conf=50) -> dict:
        result = [0] * len(self.words)
        for words in self.words {
            
        }

    
    def test(self, label, results, min_conf=0) -> dict:
        bow_representation = self.search(results, min_conf)
        print("Label: " + str(label))
        print(bow_representation)
        print("")
        return bow_representation
          
    def getDictionary(self) -> dict:
        return self.words
    


if __name__ == "__main__":
    bow = BOW('bow/BOW.txt')
    