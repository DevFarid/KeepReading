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

    @staticmethod
    def lenovo_test():
        LENOVO = [
            "4421225", "4421226", "4421227", "4421228",
            "4421320", "4421321", "4421322", "4421323", "4421324",
            "4421425", "4421426", "4421427", "4421428", "4421429",
            "4421430", "4421431", "4421432", "4421433", "4421434",
            "4421635", "4421636"
        ]
        LENOVO_CV2 = BOW_TESTER.createCVImageArray(LENOVO)
        BOW_TESTER.multiThreadedTest(LENOVO, LENOVO_CV2)

    @staticmethod
    def seagate_test():
        SEAGATE = [
            "4421637", "4421648", "4421649", "4421650", "4421651", "4421652",
            "4421653", "4421654", "4421655", "4421656", "4421657", "4421658", "4421659", 
            "4421660", "4421661", "4421662", "4421663", "4421664"
        ]
        SEAGATE_CV2 = BOW_TESTER.createCVImageArray(SEAGATE)
        BOW_TESTER.multiThreadedTest(SEAGATE, SEAGATE_CV2)

    @staticmethod
    def intel_test():
        INTEL = [
            "4421604", "4421605", "4421605", "4421606", 
            "4421607", "4421608", "4421610", "4421611"
        ]
        INTEL_CV2 = BOW_TESTER.createCVImageArray(INTEL)
        BOW_TESTER.multiThreadedTest(INTEL, INTEL_CV2)

    @staticmethod
    def hgst_test():
        HGST = [
            "4421460", "4421461", "4421462", 
            "4421463", "4421464", "4421465"
        ]
        HGST_CV2 = BOW_TESTER.createCVImageArray(HGST)
        BOW_TESTER.multiThreadedTest(HGST, HGST_CV2)

    @staticmethod
    def ibm_test():
        IBM = [
            "4421634", "4421643", "4421644", "4421645"
        ]
        IBM_CV2 = BOW_TESTER.createCVImageArray(IBM)
        BOW_TESTER.multiThreadedTest(IBM, IBM_CV2)

    @staticmethod
    def hp_test():
        HP = [
            "4421196", "4421197", "4421198", "4421199", "4421200", "4421202",
            "4421203", "4421204", "4421205", "4421206", "4421207", "4421208",
            "4421453", "4421454", "4421455", "4421456", "4421457", "4421458",
            "4421459", "4421503", "4421504", "4421505", "4421506", "4421507",
            "4421508", "4421509", "4421510", "4421512", "4421638", "4421639",
            "4421640", "4421641", "4421642"
        ]
        HP_CV2 = BOW_TESTER.createCVImageArray(HP)
        BOW_TESTER.multiThreadedTest(HP, HP_CV2)

    @staticmethod
    def dell_test():
        DELL = [
            "4421184", "4421187", "4421201", "4421214", "4421221", "4421237",
            "4421494", "4421308", "4421502", "4421528", "4421614"
        ]
        DELL_CV2 = BOW_TESTER.createCVImageArray(DELL)
        BOW_TESTER.multiThreadedTest(DELL, DELL_CV2)

if __name__ == "__main__":
    pass
    # 
    # The code below is used to run the images on OCR and get an output to then find the popular strings to identify drives via BOWs.
    # Feel free to enable each section to run different categories of drives.
    # 
    # BOW_TESTER.lenovo_test()
    # BOW_TESTER.seagate_test()
    # BOW_TESTER.intel_test()
    # BOW_TESTER.hgst_test()
    # BOW_TESTER.ibm_test()
    # BOW_TESTER.hp_test()
    # BOW_TESTER.dell_test()