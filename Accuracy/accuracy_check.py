import argparse
import cv2
from pytesseract import *
import pytesseract
import zxingcpp
import csv
import yaml

from queue import *
from threading import Thread

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--images",
				required=False,
				help="path to folder with input images")
ap.add_argument("-c", "--csv",
                required=False,
				help="path to csv file ")
ap.add_argument("-t", "--threads",
                required=False,
                type=int,
                default=10,
                help="number of threads to use")
ap.add_argument("-r", "--results",
                required=False,
                help="yaml file where failures are stored")
ap.add_argument("-f", "--file",
                required=False,
                help="input file with list of image names to test for further processing")
ap.add_argument("-b", "--barcode_removal",
                required=False,
                type=bool,
                default=False,
                help="true if barcodes should be removed, false otherwise")
args = vars(ap.parse_args())

if args['images'] is not None:
    if args['csv'] is None or args['results'] is None:
        print("CSV AND RESULTS REQUIRED")
        exit(1)
elif args['file'] is not None:
    if args['csv'] is None or args['results'] is None:
        print("CSV AND RESULTS REQUIRED")
        exit(1)
else:
    print("NO IMAGE PATH OR FILE PROVIDED.")
    exit(1)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
d_queue = Queue()
r_queue = Queue()
dataArr = []
num_threads = args['threads']

table_data = []

with open(args['csv']) as csvfile:
    tablereader = csv.reader(csvfile)
    for row in tablereader:
        table_data.append(row.copy())

table_data = table_data[1:]

#this goes through every image provided in the args['images'] folder
if args['file'] is None:
    for row in table_data:
        print("Loading image" + row[1])
        dataArr.append([row[1], cv2.cvtColor(cv2.imread(args['images'] + "\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])
        d_queue.put([row[1], cv2.cvtColor(cv2.imread(args['images'] + "\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])

#this goes through images only provided in the yaml file
else:
    with open(args["file"]) as ymlfile:
        yml_reader = yaml.safe_load(ymlfile)

    for name in yml_reader:
        print("Loading image" + row[1])
        dataArr.append([name, cv2.cvtColor(cv2.imread(args['images'] + "\\" + name + ".jpg"), cv2.COLOR_BGR2RGB)])
        d_queue.put([name, cv2.cvtColor(cv2.imread(args['images'] + "\\" + name + ".jpg"), cv2.COLOR_BGR2RGB)])

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
        results = [thing[0], process_image(image), image]
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
print("Failures:")
for failure in failures:
    print("PID: ", failure[0])
    print("Text: ", failure[1])

print("Accuracy: ", (len(successes) / (len(successes) + len(failures))))

with open(args['file'], 'w') as resultfile:
    yaml.dump(failures, resultfile)