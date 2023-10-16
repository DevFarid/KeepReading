import cv2
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove

DELL_type1_images = [cv2.imread("data\\4421175.jpg"), cv2.imread("data\\4421195.jpg"), cv2.imread("data\\4421177.jpg"), cv2.imread("data\\4421183.jpg")]
DELL_type1_images_codes = [[],[],[],[]]

SEAGATE_images = [cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421646.jpg"), cv2.imread("data\\4421647.jpg"), cv2.imread("data\\4421649.jpg")]
SEAGATE_images_codes = [[],[],[],[]]

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