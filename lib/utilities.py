import numpy as np

## CONSTANTS ##
class ConstantNames():
    #algorithms
    KNEAREST = "KNearest"

    #data reps
    BOW = "BOW"
    BWHIST = "BWHist"

### CONVERTS A CATEGORY TO A ONE-HOT ###
def convert_one_hot(label: str, label_dict: dict):
    if label in list(label_dict.keys()):
        return label_dict[label], label_dict
    elif len(list(label_dict.keys())) == 0:
        label_dict[label] = [1]
        return [1], label_dict
    else:
        fixed_dict = {}
        opt_length = 0
        for key in list(label_dict.keys()):
            fixed_dict[key] = np.append(label_dict[key], [0])
            opt_length = len(fixed_dict[key])
        fixed_dict[label] = np.append(np.zeros((opt_length - 1,)), [1])
        return fixed_dict[label], fixed_dict

### CONVERTS ALL CATEGORIES TO ONE HOTS ###
def convert_all_to_one_hots(labels):
    one_hot_dict = {}

    for label in labels:
        _, one_hot_dict = convert_one_hot(label, one_hot_dict)

    return one_hot_dict, np.array([one_hot_dict[l] for l in labels])