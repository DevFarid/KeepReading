from concurrent.futures import ThreadPoolExecutor
from functools import partial
from bow import *
from OCR import *
from preprocessing import *

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
    def process_image(filenames, cv2_images, index):
        print(f"Testing Image: {filenames[index]}.jpg\n", OCR.getResults(cv2_images[index])["text"], "\n")

    @staticmethod
    def multiThreadedTest(imgNameArr, cv2ImgObjArr):
        x = partial(BOW_TESTER.process_image, imgNameArr, cv2ImgObjArr)

        with ThreadPoolExecutor() as executor:
            executor.map(x, range(len(imgNameArr)))

if __name__ == "__main__":
    # LENOVO = [
    #     "4421225", "4421226", "4421227", "4421228",
    #     "4421320", "4421321", "4421322", "4421323", "4421324",
    #     "4421425", "4421426", "4421427", "4421428", "4421429",
    #     "4421430", "4421431", "4421432", "4421433", "4421434",
    #     "4421635", "4421636"
    # ]
    # LENOVO_CV2 = BOW_TESTER.createCVImageArray(LENOVO)
    # BOW_TESTER.multiThreadedTest(LENOVO, LENOVO_CV2)

    
    # SEAGATE = [
    #     "4421637", "4421648", "4421649", "4421650", "4421651", "4421652",
    #     "4421653", "4421654", "4421655", "4421656", "4421657", "4421658", "4421659", 
    #     "4421660", "4421661", "4421662", "4421663", "4421664"
    # ]
    # SEAGATE_CV2 = BOW_TESTER.createCVImageArray(SEAGATE)
    # BOW_TESTER.multiThreadedTest(SEAGATE, SEAGATE_CV2)

    # INTEL = [
    #     "4421604", "4421605", "4421605", "4421606", 
    #     "4421607", "4421608", "4421610", "4421611"
    # ]
    # INTEL_CV2 = BOW_TESTER.createCVImageArray(INTEL)
    # BOW_TESTER.multiThreadedTest(INTEL, INTEL_CV2)

    HGST = [
        "4421460", "4421461", "4421462", 
        "4421463", "4421464", "4421465"
    ]
    HGST_CV2 = BOW_TESTER.createCVImageArray(HGST)
    BOW_TESTER.multiThreadedTest(HGST, HGST_CV2)