import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocess
import cv2
from skimage.filters import threshold_otsu
import csv
import yaml
import h5py

from queue import *
from threading import Thread

class TrainingRepresentation():
    def represent_data(im, param: dict):
        pass
class BWHistogram(TrainingRepresentation):
    def represent_data(self, image, parameters: dict = {}):
        def remove_white(arr):
            removed_whitespace = []
            for val in arr:
                if val > 0:
                    removed_whitespace.append(val)
            return np.array(removed_whitespace)

        def regularize(arr):
            means = []
            completed = list(arr).copy()
            for i in range(len(arr) - 1):
                means.append((arr[i] + arr[i + 1]) / 2)
            
            i = 0
            index = 1
            while len(completed) < parameters['optimal_size']:
                if i < len(means):
                    completed.insert(index, means[i])
                else:
                    break
                index += 2
                i += 1

            if len(completed) < parameters['optimal_size']:
                completed = regularize(completed)

            while len(completed) > parameters['optimal_size']:
                completed.pop()

            return completed
 
        feature_vector = []
        # 1) preprocess
        b_w_image = Preprocess.gaussian_blur(Preprocess.to_bw(Preprocess.remove_bg(image)), 1)
        processed_image = Preprocess.to_binary(b_w_image, threshold_otsu(b_w_image))
        # 2) convert to histogram array
        sums = np.sum(processed_image, axis=0)
        reg_sum = sums / max(sums)
        feature_vector_p = regularize(remove_white(reg_sum))

        # 3) smooth?
        for num in feature_vector_p:
            feature_vector.append(float(num))
        return feature_vector

