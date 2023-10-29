import pymongo
from datetime import datetime

class OcrDatabase:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["keepreading"]
        self.col = self.db["ocrtest"]
        self.col.find().sort

class Api:
    @staticmethod
    def insert(document, collection):
        return collection.insert_one(document)
    @staticmethod
    def find(filter: tuple, collection):
        return collection.find(filter)
    @staticmethod
    def find_one(filter: tuple, collection):
        return collection.find_one(filter)
    @staticmethod
    def find_most_recent(collection):
        return collection.find().sort("datetime", -1).limit(1)

class OcrResult:
    def __init__(self, pid: int, auditId: int, manufacturer: str, model: str, serialNumber: str, confidence: str, userReported: bool, datetime: datetime):
        self.pid = pid
        self.auditId = auditId
        self.manufacturer = manufacturer
        self.model = model
        self.serialNumber = serialNumber
        self.confidence = confidence
        self.userReported = userReported
        self.datetime = datetime

# Sample client creation and db insertion, finding list of matching documents, and finding first matching document
# client = OcrDatabase()
# ocr = OcrResult(1234567, 12345678, "MANUFACTURER", "MODEL", "SN", "100%", True, datetime.now()).__dict__
# testPost = Api.insert(ocr, client.col)

# filter = {'pid':1234567}
# testGetList = Api.find(filter, client.col)
# testGetOne = Api.find_one(filter, client.col)

# print(testPost.inserted_id)
# print("/////////////////////////////")
# print(testGetOne)
# print("/////////////////////////////")
# for test in testGetList:
#     print(test)