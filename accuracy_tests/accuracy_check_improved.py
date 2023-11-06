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
ser_model_dict = {}

num_threads = args['threads']

table_data = []


#this goes through every image provided in the args['images'] folder
if args['file'] is None:
    with open(args['csv']) as csvfile:
        tablereader = csv.reader(csvfile)
        for row in tablereader:
            table_data.append(row.copy())

    table_data = table_data[1:]
    for row in table_data:
        print("Loading image" + row[1])
        dataArr.append([row[1], cv2.cvtColor(cv2.imread(args['images'] + "\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])
        ser_model_dict[row[1]] = {'SerialNumber': row[4], 'Model':row[3]}
        d_queue.put([row[1], cv2.cvtColor(cv2.imread(args['images'] + "\\" + row[1] + ".jpg"), cv2.COLOR_BGR2RGB)])

#this goes through images only provided in the yaml file
else:
    with open(args['csv']) as csvfile:
        tablereader = csv.DictReader(csvfile)
        for row in tablereader:
            ser_model_dict[row['PID']] = {'SerialNumber': row['SerialNumber'], 'Model':row['Model']}

    with open(args["file"]) as ymlfile:
        yml_reader = yaml.safe_load(ymlfile)

    for name in yml_reader:
        print("Loading image" + name)
        dataArr.append([name, cv2.cvtColor(cv2.imread(args['images'] + "\\" + name + ".jpg"), cv2.COLOR_BGR2RGB)])
        d_queue.put([name, cv2.cvtColor(cv2.imread(args['images'] + "\\" + name + ".jpg"), cv2.COLOR_BGR2RGB)])

for _ in range(num_threads):
    d_queue.put(None)

def process_image(image):
    results = pytesseract.image_to_string(image, output_type=Output.DICT)
    return results.copy()

def process_image_barcode_removal(image):
    results = zxingcpp.read_barcodes(image)
    for result in results:
        t = str(result.position)[:-1]
        t = [list(map(int, x.split("x"))) for x in t.split(" ")]

        coords = {
            "top_right": {
                "x": t[0][0],
                "y": t[0][1],
            },
            "bottom_right": {
                "x": t[1][0],
                "y": t[1][1],
            },
            "bottom_left": {
                "x": t[2][0],
                "y": t[2][1],
            },
            "top_left": {
                "x": t[3][0],
                "y": t[3][1],
            },
	    }
        image = cv2.rectangle(image, (coords["top_left"]["x"],coords["top_left"]["y"]),(coords["bottom_right"]["x"],coords["bottom_right"]["y"]), (0, 0, 255), -1)
    return pytesseract.image_to_string(image, output_type=Output.DICT)

def image_processing_worker():
    while True:
        thing = d_queue.get()
        if thing is None:
            break
        image = thing[1]
        if args["barcode_removal"]:
            results = [thing[0], process_image_barcode_removal(image), image]
        else:
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
ser = [[],[]]
models = [[],[]]
for entry in processed_results:
    if entry[0] in entry[1]["text"]:
        successes.append(entry)
    else:
        failures.append(entry)

    if ser_model_dict[entry[0]]['SerialNumber'].upper() in entry[1]["text"]:
        ser[0].append(entry)
    else:
        ser[1].append(entry)
    
    if ser_model_dict[entry[0]]['Model'].upper() in entry[1]["text"]:
        models[0].append(entry)
    else:
        models[1].append(entry)

print("Failures:")
for failure in failures:
    print("PID: ", failure[0])
    print("Text: ", failure[1])

print("PID Accuracy: ", (len(successes) / (len(successes) + len(failures))))
print("SER Accuracy: ", (len(ser[0]) / (len(ser[0]) + len(ser[1]))))
print("MOD Accuracy: ", (len(models[0]) / (len(models[0]) + len(models[1]))))

print("PID Success Length: ", len(successes))
print("PID Failure Length: ", len(failures))

with open(args['results'], 'w') as resultfile:
    yaml.dump([failure[0] for failure in failures], resultfile)