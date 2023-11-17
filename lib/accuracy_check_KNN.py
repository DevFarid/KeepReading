import csv

results = []
with open("test_results.txt") as txtfile:
    while True:
        line = txtfile.readline()
        if not line:
            break
        results.append(line.split(sep=','))

ser_model_dict = {}
with open("..\\data\\15021026 1 fixed.csv", "r") as csvfile:
    tablereader = csv.DictReader(csvfile)
    for row in tablereader:
        ser_model_dict[row['PID']] = {'SerialNumber': row['SerialNumber'], 'Model':row['Model']}

results = [[result.strip() for result in entry] for entry in results]
for i in range(len(results)):
    results[i][0] = results[i][0][9:16]

accuracies = [[0,0],[0,0],[0,0]]

for result in results:
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

print(f'PID Accuracy: {(accuracies[0][0] / (accuracies[0][0] + accuracies[0][1])) * 100}% Correct out of {accuracies[0][0] + accuracies[0][1]}')
print(f'S/N Accuracy: {(accuracies[1][0] / (accuracies[1][0] + accuracies[1][1])) * 100}% Correct out of {accuracies[1][0] + accuracies[1][1]}')
print(f'Model Accuracy: {(accuracies[2][0] / (accuracies[2][0] + accuracies[2][1])) * 100}% Correct out of {accuracies[2][0] + accuracies[2][1]}')