from concurrent.futures import ThreadPoolExecutor
from bow import *
from OCR import *

class BOW_TESTER():
    """
        Bag of Words Tester
        Created by Farid Kamizi on 10/19/2023.
        Tests the accuracy of BOW algorithm for given training data.
    """     
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_image(strFileLoc: str):
        return cv2.imread("data\\" + strFileLoc + ".jpg")

    @staticmethod
    def createCVImageArray(strImages: list) -> list:
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(BOW_TESTER.load_image, strImages))
        return data
    
    @staticmethod
    def getTrainingDataDict() -> list:
        return ["DELL", "HP ", "LENOVO", "IBM", "HGST", "SAMSUNG", "INTEL", "SEAGATE"]
    
    @staticmethod
    def test_images(key, image):
        ocrResults = OCR.getResults(image)
        evalDict = bow.test(key, ocrResults)
        
        if key not in evalDict or evalDict[key] <= 0:
            return 0
        else:
            for k, value in evalDict.items():
                if k != key and value > 0:
                    return 0
            return 1

if __name__ == "__main__":
    DELL = BOW_TESTER.createCVImageArray(["4421633", "4421582", "4421488", "4421442"])
    HP = BOW_TESTER.createCVImageArray(["4421207", "4421202", "4421185", "4421642"])
    LENOVO = BOW_TESTER.createCVImageArray(["4421226", "4421278", "4421322", "4421297"])
    IBM = BOW_TESTER.createCVImageArray(["4421430", "4421634", "4421314", "4421315"])
    HGST = BOW_TESTER.createCVImageArray(["4421460", "4421461", "4421462", "4421463"])
    SAMSUNG = BOW_TESTER.createCVImageArray(["4421603"])
    INTEL = BOW_TESTER.createCVImageArray(["4421604", "4421605", "4421606", "4421611"])
    SEAGATE = BOW_TESTER.createCVImageArray(["4421664", "4421647", "4421646", "4421637"])
    # EDGE_CASES = BOW_TESTER.createCVImageArray(["4421201", "4421449", "4421311"])

    concat = dict()
    concat["DELL"] = DELL
    concat["HP"] = HP
    concat["LENOVO"] = LENOVO
    concat["IBM"] = IBM
    concat["HGST"] = HGST
    concat["SAMSUNG"] = SAMSUNG
    concat["INTEL"] = INTEL
    concat["SEAGATE"] = SEAGATE

    bow = BOW()
    bow.setDictionary(BOW_TESTER.getTrainingDataDict())

    correctClassification = 0
    total_tests = 0

    with ThreadPoolExecutor() as executor:
        for key, value in concat.items():
            results = list(executor.map(BOW_TESTER.test_images, [key] * len(value), value))
            correctClassification += sum(results)
            total_tests += len(value)

    print(f"Correctly Labeled: {correctClassification}\nTotal Images: {total_tests}\nAccuracy: {correctClassification/total_tests*100}%")