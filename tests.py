import unittest
import re
from datetime import datetime
from api import OcrDatabase, Api, OcrResult

class Tests(unittest.TestCase):
    def testA(self):
        client = OcrDatabase()
        assert re.search(r"\s*(?=\d{2}(?:\d{2})?-\d{1,2}-\d{1,2}\b)", str(Api.find_most_recent(client.col).next()['datetime'])) != None, "testA failed"

if __name__ == "__main__":
    unittest.main() # run all tests