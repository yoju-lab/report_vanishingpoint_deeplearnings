import os
import pandas as pd
columns_list = ['file_abspath', 'file_name', 'height', 'width']
def drop_images_infromation(images_informations):
    # make pandas dataset
    images_df = pd.DataFrame(images_informations, columns=columns_list)
    # print(images_df)
    print(images_df.describe(include='all'))
    images_df.to_csv('./images_informations.csv')
    return images_df

def get_images_information(root_directory=r'photos/'):
    # folder path
    dir_path = root_directory

    # list to store files
    images_informations = []
    images_sizes = []
    read_count = 0
    import matplotlib.pyplot as plt
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        current_path = os.path.join(dir_path,path)
        for file in os.listdir(current_path):
            file_abspath = os.path.abspath(os.path.join(current_path, file))
            read_count += 1
            if os.path.isfile(file_abspath):
                image = plt.imread(file_abspath)
                shape = image.shape
                images_sizes.append([shape[0], shape[1]])
                images_informations.append([file_abspath, file, shape[0], shape[1]])
    # print(read_count)
    images_df = drop_images_infromation(images_informations)
    return (images_sizes, images_df)   

import numpy as np
def get_min_size(images_sizes):
    np_images_sizes = np.asarray(images_sizes)
    min_height = np_images_sizes[:,0].min(axis=0)
    min_width = np_images_sizes[:,1].min(axis=0)
    print('images_sizes min_height : ${}, min_width : ${}'.format(min_height,min_width))
    return min_height, min_width

import cv2
def resize_image(row, min_height, min_width):
    # read the image
    image = cv2.imread(row['file_abspath'])
    pass
    # covert gray scale

    # get ratio of the image
    # ratio = 600.0 / image.shape[1]
    # dim = (600, int(image.shape[0] * ratio))
    
    # # perform the actual resizing of the image
    # resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Resized (Width)", resized)    
    
def resize_images(images_df, min_height, min_width):
    searchs = ['buildings12.jpeg', 'buildings45.jpeg']
    conditions = (images_df['file_name'] == searchs[0]) | (images_df['file_name'] == searchs[1])
    images_df = images_df[conditions]
    
    images_df = images_df.reset_index()  # make sure indexes pair with number of rows

    train_path = 'photos/train_images'
    for index, row in images_df.iterrows():
        # print(row['c1'], row['c2'])
        resize_image(row, min_height, min_width)
        # save_path = os.path.join(train_path, file_name)                     