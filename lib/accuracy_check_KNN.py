import csv
import argparse

training_results = []
with open("training_results.txt") as txtfile:
    while True:
        line = txtfile.readline()
        if not line:
            break
        training_results.append(line.split(sep=','))

test_results = []
with open("test_results.txt") as txtfile:
    while True:
        line = txtfile.readline()
        if not line:
            break
        test_results.append(line.split(sep=','))

ap = argparse.ArgumentParser()
ap.add_argument("csv",
                help="path to formatted csv file")
ap.add_argument("--extension",
                required=False,
                help="file extension to load in",
                default=".jpg")
args = vars(ap.parse_args())

ser_model_dict = {}
with open(args["csv"], "r") as csvfile:
    tablereader = csv.DictReader(csvfile)
    for row in tablereader:
        ser_model_dict[row['PID']] = {'SerialNumber': row['SerialNumber'], 'Model':row['Model']}

training_results = [[result.strip() for result in entry] for entry in training_results]
for i in range(len(training_results)):
    training_results[i][0] = training_results[i][0][len(args["csv"]) + 1:len(training_results[i][0]) - len(args["extension"])]

accuracies = [[0,0],[0,0],[0,0]]

for result in training_results:
    if result[0] == result[1]:
        accuracies[0][0] += 1
    else:
        accuracies[0][1] += 1
    if ser_model_dict[result[0]]['SerialNumber'] == result[2]:
        accuracies[1][0] += 1
    else:
        accuracies[1][1] += 1
    if ser_model_dict[result[0]]['Model'] in result[3]:
        accuracies[2][0] += 1
    else:
        accuracies[2][1] += 1

print(f'\nTraining Results:')
print(f'PID Accuracy: {(accuracies[0][0] / (accuracies[0][0] + accuracies[0][1])) * 100}% Correct out of {accuracies[0][0] + accuracies[0][1]}')
print(f'S/N Accuracy: {(accuracies[1][0] / (accuracies[1][0] + accuracies[1][1])) * 100}% Correct out of {accuracies[1][0] + accuracies[1][1]}')
print(f'Model Accuracy: {(accuracies[2][0] / (accuracies[2][0] + accuracies[2][1])) * 100}% Correct out of {accuracies[2][0] + accuracies[2][1]}')

test_results = [[result.strip() for result in entry] for entry in test_results]
for i in range(len(test_results)):
    test_results[i][0] = test_results[i][0][len(args["csv"]) + 1:len(test_results[i][0]) - len(args["extension"])]

accuracies = [[0,0],[0,0],[0,0]]

for result in test_results:
    if result[0] == result[1]:
        accuracies[0][0] += 1
    else:
        accuracies[0][1] += 1
    if ser_model_dict[result[0]]['SerialNumber'] == result[2]:
        accuracies[1][0] += 1
    else:
        accuracies[1][1] += 1
    if ser_model_dict[result[0]]['Model'] in result[3]:
        accuracies[2][0] += 1
    else:
        accuracies[2][1] += 1

print(f'\nTest Results:')
print(f'PID Accuracy: {(accuracies[0][0] / (accuracies[0][0] + accuracies[0][1])) * 100}% Correct out of {accuracies[0][0] + accuracies[0][1]}')
print(f'S/N Accuracy: {(accuracies[1][0] / (accuracies[1][0] + accuracies[1][1])) * 100}% Correct out of {accuracies[1][0] + accuracies[1][1]}')
print(f'Model Accuracy: {(accuracies[2][0] / (accuracies[2][0] + accuracies[2][1])) * 100}% Correct out of {accuracies[2][0] + accuracies[2][1]}')