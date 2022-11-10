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
    print('images_sizes min_height : {}, min_width : {}'.format(min_height,min_width))
    return min_height, min_width

import cv2
def center_crop(img, min_height, min_width):

    h, w, c = img.shape

    # if set_size > min(h, w):
    #     return img

    crop_width = min_width
    crop_height = min_height

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def resized_image(image, min_height, min_width):
    # rotate vertical image to horizontal
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    # get ratio of the image
    ratio = min_height / image.shape[0]
    dim = (int(image.shape[1] * ratio), min_height)
    
    # # perform the actual resizing of the image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    resized_image = center_crop(resized_image, min_height, min_width)
    return resized_image

def read_any_image(file_abspath):
    try:
        # gif 처리
        if str(file_abspath).lower().endswith('.gif'):
            gif = cv2.VideoCapture(file_abspath)
            ret, frame = gif.read()  # ret=True if it finds a frame else False.
            if ret:
                return frame
        else:
            return cv2.imread(file_abspath)
    except Exception as e:
        print(e)
        return None

def resize_image(row, min_height, min_width):
    # read the image
    image = read_any_image(row['file_abspath'])
    # read image as None
    if isinstance(image,type(None)):
        print('image none {}'.format(row['file_abspath']))
    pass

    resizing_image = resized_image(image, min_height, min_width)
    # covert gray scale
    image_gray = convert_to_gray(resizing_image)

    # cv2.imshow("Resized (Width)", image_gray)
    # cv2.waitKey()
    pass
    return image_gray

def write_any_image(file_name, image_gray):
    train_path = 'pre_images'

    try:
        file_name = file_name.split('.')[0] + '.png'

        file_name = 'resized_' + file_name
        save_path = os.path.join(train_path, file_name)
        cv2.imwrite(save_path, image_gray)  
    except Exception as e:
        print(e)
        return None


def resize_images(images_df, min_height, min_width):
    # short test
    # searchs = ['buildings12.jpeg', 'buildings45.jpeg']
    # conditions = (images_df['file_name'] == searchs[0]) | (images_df['file_name'] == searchs[1])
    # images_df = images_df[conditions]
    
    images_df = images_df.reset_index()  # make sure indexes pair with number of rows

    for index, row in images_df.iterrows():
        image_gray = resize_image(row, min_height, min_width)
        write_any_image(row['file_name'], image_gray)

    print('index : {}'.format(index+1))         