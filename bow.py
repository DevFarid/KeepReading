import cv2
from bow_tester import *
from OCR import *
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import re
from cycler import cycler

class BOW(TrainingRepresentation):
    """
        Bag of Words
        Created by Farid Kamizi on 10/19/2023.
        This class represents an algorithm called `Bag of Words`.
        Given a "document", our goal is to search for interested words in that document.
    """     
    def __init__(self, file) -> None:
        self.bag_of_words = dict()
        self.loadWordsFromFile(file)
        s = "\n"
        for key in self.bag_of_words:
            s += f"{key}: {self.bag_of_words[key]}\n"
        print(s)

    def getAllWords(self, dict) -> list:
        return [word for words in dict.values() for word in words]

    # TODO: load words from file.
    def loadWordsFromFile(self, file) -> None:
        lastKey = None
        with open(file, "r") as f:
            for line in (l.replace("\n", "") for l in f if len(l.strip()) != 0):
                match = re.search(r'^\/\/\s(\w+)', line) 
                if match:
                    lastKey = match.group(1)
                    self.bag_of_words[lastKey] = []
                else:
                    self.bag_of_words[lastKey].append(line)

    def setDictionary(self, newDict) -> None:
        self.bag_of_words = newDict

    def addWord(self, label, words) -> None:
        self.bag_of_words[label] = words

    def removeLabel(self, label) -> None:
        return self.bag_of_words[label] if label in self.bag_of_words.keys else None
    
    # runs bag of words on OCR results.
    def search(self, img) -> dict:
        bow_vector = {key: [0] * len(words) for key, words in self.bag_of_words.items()}
        ocrResults = [string.lower() for string in OCR.getResults(img)["text"]]

        # false flagging removal detection
        for i, word in enumerate(ocrResults):
            if word in {"manufacture", "mfg"} and i + 2 < len(ocrResults) and ocrResults[i + 1] == "by":
                next_word = ocrResults[i + 2]
                if any(next_word in words for words in self.bag_of_words.values()):
                    ocrResults.pop(i + 2)

        for key, words in self.bag_of_words.items():
            for i, word in enumerate(words):
                if word in ocrResults:
                    bow_vector[key][i] = 1

        return bow_vector


    def reduce(self, R: list, N: int) -> list:
        result = []
        i = 0
        while i < len(R):
            sum = 0
            for j in range(N):
                sum += R[i + j]
            result.append(sum)
            i += N
        return result

    def test(self, label, results, min_conf=0) -> dict:
        bow_representation = self.search(results, min_conf)
        print("Label: " + str(label))
        print(bow_representation)
        print("")
        return bow_representation
    
    def represent_data(im, param: dict):
        return list(BOW().search(im).values())
          
    def getDictionary(self) -> dict:
        return self.bag_of_words
    
    def plot_results(self, result, label):
        labels = self.getAllWords(self.bag_of_words)
        values = [result[key][i] for key, words in self.bag_of_words.items() for i in range(len(words))]

        ax = plt.gca()  # Get the current Axes instance
        bars = plt.bar(labels, values)
        plt.ylabel('Presence (0 or 1)')
        plt.title(f'BOW Results for {label}')
        plt.xticks(rotation=45, ha='right')
        
        color_cycler = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
        colors = iter(color_cycler)
        
        # Set specific tick label colors
        for key, words in self.bag_of_words.items():
            color = next(colors)['color']
            for word in words:
                if word in labels:
                    index = labels.index(word)
                    bars[index].set_color(color)  # Set bar color
                    ax.get_xticklabels()[index].set_color(color)  # Set tick label color
        
        plt.ylim(0, 1)
        plt.tight_layout()

    def plot_all(self, data, results, images):
        # Create a figure with subplots using gridspec
        plt.figure(figsize=(15, 10))
        gs = GridSpec(len(data), 2, width_ratios=[1, 3])  # adjust width_ratios as needed

        for i in range(len(data)):
            # Plot images
            ax1 = plt.subplot(gs[i, 0])
            ax1.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert image from BGR to RGB
            ax1.axis('off')  # Turn off axis
            
            # Plot results
            ax2 = plt.subplot(gs[i, 1])
            plt.sca(ax2)  # Set the current Axes instance to ax2
            bow.plot_results(results[i], f"{data[i]}.jpg")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    bow = BOW("bow/bow.txt")

    dell_data = [4421201, 4421221, 4421511, 4421535, 4421625]
    lenovo_data = [4421225, 4421228, 4421324, 4421517, 4421524]
    
    def openCV(str):
        return cv2.imread(f"data\\{str}.jpg")

    with ThreadPoolExecutor() as executor:
        dell_cv2 = list(executor.map(openCV, dell_data))
        lenovo_cv2 = list(executor.map(openCV, lenovo_data))

    # Now you have the images, you can run the BOW search
    with ThreadPoolExecutor() as executor:
        dell_results = list(executor.map(bow.search, dell_cv2))
        lenovo_results = list(executor.map(bow.search, lenovo_cv2))

    bow.plot_all(dell_data, dell_results, dell_cv2)
    bow.plot_all(lenovo_data, lenovo_results, lenovo_cv2)