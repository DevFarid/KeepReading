import pymongo
import json

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["keepreading"]
col = db["ocrtest"]

testData = '{ "pid":4421175, "auditId":15021026, "manufacturer":"DELL", "model":"AL15SEB060NY", "serialNumber":"Y8D0A04LFQWF", "confidence":"80%", "userReported":true }'
print(json.loads(testData))

testJson = json.loads(testData)
testPost = col.insert_one(testJson)

print(testPost.inserted_id)