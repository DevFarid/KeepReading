import cv2
class BOW:
    """
        Bag of Words
        Created by Farid Kamizi on 10/19/2023.
        This class represents an algorithm called `Bag of Words`.
        Given a "document", our goal is to search for interested words in that document.
    """     
    def __init__(self) -> None:
        self.dictionary = list()

    # TODO: load words from file.
    def loadWordsFromFile(self):
        pass 

    def setDictionary(self, newDict) -> None:
        self.dictionary = newDict

    def addWord(self, word) -> None:
        self.dictionary.append(word)

    def removeWord(self, word) -> None:
        return self.dictionary.remove(word) if self.dictionary.__contains__(word) else None
    
    # runs bag of words on OCR results.
    def search(self, results, min_conf=0) -> dict:
        bow_representation = dict()
        if "text" in results:
            for word in self.dictionary:
                bow_representation[word] = 0
            for i in range(0, len(results["text"])):
                text = results["text"][i]
                conf = int(results["conf"][i])
                
                if conf >= min_conf:
                    if self.dictionary.__contains__(text):
                        bow_representation[text] = bow_representation[text] + 1
        return bow_representation
    
    def test(self, label, results, min_conf=0) -> dict:
        bow_representation = self.search(results, min_conf)
        print("Label: " + str(label))
        print(bow_representation)
        print("")
        return bow_representation
          
    def getDictionary(self) -> dict:
        return self.dictionary
    

