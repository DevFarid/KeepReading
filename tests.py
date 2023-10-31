import unittest
import re
from datetime import datetime
from api import OcrDatabase, Api, OcrResult

class Tests(unittest.TestCase):
    def testA(self):
        client = OcrDatabase()
        assert re.search(r"\s*(?=\d{2}(?:\d{2})?-\d{1,2}-\d{1,2}\b)", str(Api.find_most_recent(client.col).next()['datetime'])) != None, "testA failed"
    def testB(self):
        client = OcrDatabase()
        ocr = OcrResult(1234567, 12345678, "MANUFACTURER", "MODEL", "SN", "100%", True, datetime.now()).__dict__
        testPost = Api.insert(ocr, client.col)
        assert testPost.inserted_id != None
    def testC(self):
        client = OcrDatabase()
        testGet = Api.find({}, client.col)
        assert len(list(testGet)) > 0
    def testD(self):
        client = OcrDatabase()
        testGet = Api.find_one({}, client.col)
        assert testGet != None

if __name__ == "__main__":
    unittest.main() # run all tests