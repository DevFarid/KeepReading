import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocess
import cv2

DELL_type1_images = [cv2.imread("data\\4421175.jpg"), cv2.imread("data\\4421195.jpg"), cv2.imread("data\\4421177.jpg"), cv2.imread("data\\4421183.jpg")]
DELL_type1_images_codes = [[],[],[],[]]

SEAGATE_images = [cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421647.jpg"), cv2.imread("data\\4421649.jpg")]
SEAGATE_images_codes = [[],[],[],[]]

def represent_data(image, parameters: dict = {}):
    def remove_white(arr):
        removed_whitespace = []
        for val in arr:
            if arr != 0:
                removed_whitespace.append(val)
        return removed_whitespace

    def regularize(arr):
        o_s = parameters['optimal_size']
        means = []
        completed = []
        for i in range(len(arr) - 2):
            means.append((arr[i + 1] + arr[i + 2]) / 2)
        
        i = 0
        alter = False
        while i < len(arr):
            if not alter:
                completed.append(arr[i])
            i += 1

        
    feature_vector = []
    # 1) preprocess
    processed_image = Preprocess.to_binary(Preprocess.gaussian_blur(Preprocess.to_bw(Preprocess.remove_bg(image)), 1), 180)
    plt.imshow(processed_image)
    plt.show()
    # 2) convert to histogram array
    sums = np.sum(processed_image, axis=0)
    reg_sum = sums / max(sums)
    feature_vector = regularize(remove_white(reg_sum))

    # 3) smooth?    
    return feature_vector

plt.imshow(DELL_type1_images[2])
plt.show()
represent_data(DELL_type1_images[2])

"""
threshold = 180
for i in range(len(DELL_type1_images)):
    DELL_type1_images[i] = 255 - cv2.cvtColor(remove(DELL_type1_images[i]), cv2.COLOR_RGB2GRAY)
    DELL_type1_images[i] = np.where(DELL_type1_images[i] < threshold, 0, 1)
    for k in range(DELL_type1_images[i].shape[1]):
        DELL_type1_images_codes[i].append(np.sum(DELL_type1_images[i][:, k]))
    DELL_type1_images_codes[i] /= np.max(DELL_type1_images_codes[i])



for i in range(len(SEAGATE_images)):
    SEAGATE_images[i] = 255 - cv2.cvtColor(remove(SEAGATE_images[i]), cv2.COLOR_RGB2GRAY)
    SEAGATE_images[i] = np.where(SEAGATE_images[i] < threshold, 0, 1)
    for k in range(SEAGATE_images[i].shape[1]):
        SEAGATE_images_codes[i].append(np.sum(SEAGATE_images[i][:, k]))
    SEAGATE_images_codes[i] /= np.max(SEAGATE_images_codes[i])

images = [DELL_type1_images, SEAGATE_images]
image_codes = [DELL_type1_images_codes, SEAGATE_images_codes]
rows = len(images)

columns = len(image_codes[0])
fig, ax = plt.subplots(rows, columns, sharex='col', sharey='row')

for row in range(rows):
    for col in range(columns):
        btm_labels = np.arange(len(image_codes[row][col]))
        ax[row,col].bar(btm_labels, image_codes[row][col])

plt.show()

avg_curves = []
for code_list in range(len(image_codes)):
    avg_curves.append(np.sum(image_codes[code_list], axis=0)/len(image_codes[code_list][0]))

rows = len(images)
fig, ax = plt.subplots(rows, 1, sharex='col', sharey='row')
for row in range(rows):
    btm_labels = np.arange(len(avg_curves[row]))
    ax[row].bar(btm_labels, avg_curves[row])

plt.show()

pass
"""