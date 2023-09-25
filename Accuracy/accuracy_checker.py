import argparse
import cv2
from pytesseract import *
import pytesseract
import csv

from queue import *
from threading import Thread, Lock

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
d_queue = Queue()
r_queue = Queue()
dataArr = []
num_threads = 10

table_data = []

with open('Training Data\\15021026 1.csv') as csvfile:
    tablereader = csv.reader(csvfile)
    for row in tablereader:
        table_data.append(row.copy())

table_data = table_data[1:]
for row in table_data:
    print("Loading image" + row[1])
    dataArr.append([row[1], cv2.cvtColor(cv2.imread("Training Data\\images\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])
    d_queue.put([row[1], cv2.cvtColor(cv2.imread("Training Data\\images\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])
for _ in range(num_threads):
    d_queue.put(None)

def process_image(image):
    results = pytesseract.image_to_string(image, output_type=Output.DICT)
    return results.copy()

def image_processing_worker():
    while True:
        thing = d_queue.get()
        if thing is None:
            break
        image = thing[1]
        results = [thing[0], process_image(image)]
        r_queue.put(results)
        d_queue.task_done()
        print("PID: ", thing[0])

threads = [] 
for _ in range(num_threads):
    thread = Thread(target=image_processing_worker)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

processed_results = []
while not r_queue.empty():
    results = r_queue.get()
    processed_results.append(results)

successes = []
failures = []
for entry in processed_results:
    if entry[0] in entry[1]["text"]:
        successes.append(entry)
    else:
        failures.append(entry)
print(processed_results)

print("Accuracy: ", (len(successes) / (len(successes) + len(failures))))