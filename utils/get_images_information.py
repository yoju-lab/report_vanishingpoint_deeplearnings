import os

# folder path
dir_path = r'photos/'

# list to store files
images_informations = []
images_sizes = []
read_count = 0
# import cv2 as cv
import matplotlib.pyplot as plt
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    current_path = os.path.join(dir_path,path)
    for file in os.listdir(current_path):
        file_path = os.path.abspath(os.path.join(current_path, file))
        read_count += 1
        if os.path.isfile(file_path):
            image = plt.imread(file_path)
            shape = image.shape
            images_sizes.append([shape[0], shape[1]])
            images_informations.append([file_path, file, shape[0], shape[1]])
            
# print(images_informations)
pass
import numpy as np
np_images_sizes = np.asarray(images_sizes)
maxs = np_images_sizes.max(axis=0)
mins = np_images_sizes.min(axis=0)
print('images_sizes maxs : ${}, mins : ${}'.format(maxs,mins))
pass

# make pandas dataset
import pandas as pd
df = pd.DataFrame(images_informations, columns=['file_path', 'file_name', 'height', 'width'])
print(df)
print(read_count)