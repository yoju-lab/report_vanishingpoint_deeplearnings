import re
from random import randint, choice
import pathlib
import cv2
import numpy as np
import os
import pandas as pd
import json
configs_path = 'commons/configs.json'
configs = json.load(open(configs_path))

columns_list = ['file_abspath', 'file_name', 'height', 'width']


def re_split(value):
    result_list = re.split('[^\d\.-]+', value)
    return [result_list[1], result_list[2]]


def get_lables(data_dir):
    # read from csv
    find_vanishingpoints_path = os.path.join(configs['datasets_dir'], configs['any_informations_dir'],
                                             configs['find_vanishingpoints_csv'])
    df = pd.read_csv(find_vanishingpoints_path)
    df_split = df['vanishing_point_1']
    train_labels = list(map(re_split, df_split))

    return train_labels


def get_lables_temporaray(data_dir):
    # count with temporary labels
    count_dir = pathlib.Path(data_dir)
    image_count = len(list(count_dir.glob('*/*')))
    print(data_dir, image_count)
    train_labels = []

    for _ in range(image_count):
        x = choice([randint(9, 15), randint(21, 27), randint(1, 5)])
        y = choice([randint(1, 5), randint(9, 15), randint(21, 27)])
        train_labels.append([x, y])
        # train_labels.append([y])

    pass
    return train_labels

# 특정 사이즈 보다 작으면 
def drop_images_infromation(images_informations):
    # make pandas dataset
    images_df = pd.DataFrame(images_informations, columns=columns_list)
    # print(images_df)
    print(images_df.describe(include='all'))
    any_informations_dir = os.path.join(
        configs['datasets_dir'], configs['any_informations_dir'])
    os.makedirs(any_informations_dir, exist_ok=True)
    infor_csv_path = os.path.join(
        any_informations_dir, configs['preprocess_images_csv'])

    images_df.to_csv(infor_csv_path)
    return images_df

# 숨김 폴더를 제외
def listdir_no_hidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def load_image_paths_from_folder(folder):
    image_paths = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith((".jpeg",".jpg",".png")): # 이미지 파일 확장자에 따라 조절
                file_path = os.path.join(folder, subdir, file)
                relative_path = os.path.relpath(file_path, folder)
                image_paths.append(relative_path)
    return image_paths

def get_images_information(root_directory=r'photos/'):
    # folder path
    dir_path = root_directory

    # list to store files
    images_informations = []
    images_sizes = []
    read_count = 0
    import matplotlib.pyplot as plt

    dir_path_list = listdir_no_hidden(dir_path)
    # Iterate directory
    for path in dir_path_list:
        # check if current path is a file
        current_path = os.path.join(dir_path, path)
        current_path_list = listdir_no_hidden(current_path)

        for file in current_path_list:
            file_abspath = os.path.abspath(os.path.join(current_path, file))
            read_count += 1
            if os.path.isfile(file_abspath):
                image = plt.imread(file_abspath)
                shape = image.shape
                images_sizes.append([shape[0], shape[1]])
                images_informations.append(
                    [file_abspath, file, shape[0], shape[1]])
    # print(read_count)
    images_df = drop_images_infromation(images_informations)
    return images_df


def get_min_size(images_sizes):
    np_images_sizes = np.asarray(images_sizes)
    min_height = np_images_sizes[:, 0].min(axis=0)
    min_width = np_images_sizes[:, 1].min(axis=0)
    print('images_sizes min_height : {}, min_width : {}'.format(
        min_height, min_width))
    return min_height, min_width


def center_crop(img, min_height, min_width):

    h, w, c = img.shape

    # if set_size > min(h, w):
    #     return img

    crop_width = min_width
    crop_height = min_height

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2

    crop_img = img[mid_y - offset_y:mid_y +
                   offset_y, mid_x - offset_x:mid_x + offset_x]
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

# 이미지 종류 상관 없이 처리


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

# 경로로 image 파일 읽어 min_height, min_width 기준 사이즈 조정과 gray로 변환
def resize_image(row, min_height, min_width):
    # read the image
    image = read_any_image(row)
    # read image as None
    if isinstance(image, type(None)):
        print('image none {}'.format(row))
    pass

    resizing_image = resized_image(image, min_height, min_width)
    # covert gray scale
    image_gray = convert_to_gray(resizing_image)

    # cv2.imshow("Resized (Width)", image_gray)
    # cv2.waitKey()
    pass
    return image_gray

from datetime import datetime

def write_any_image(preprocess_images_path, file_name, image_gray):
    try:
        # 파일 이름 추출
        filename = os.path.basename(file_name)
        # 현재 시간을 시분초 형태로 가져오기
        now = datetime.now()
        prefix_file_name = '{}_{}_{}.png'.format(configs['preprocess_images_prefix']
                                           ,filename.split('.')[0]
                                           ,now.strftime('%H%M%S%f')[:12])

        save_path = os.path.join(preprocess_images_path, prefix_file_name)
        cv2.imwrite(save_path, image_gray)
    except Exception as e:
        print(e)
        return None

# 수집된 이미지를 min_height, min_width 기준으로 사이즈 조정
def resize_images(images_list, min_height, min_width):
    # short test
    # searchs = ['buildings12.jpeg', 'buildings45.jpeg']
    # conditions = (images_df['file_name'] == searchs[0]) | (images_df['file_name'] == searchs[1])
    # images_df = images_df[conditions]

    # images_df = images_df.reset_index()  # make sure indexes pair with number of rows

    save_path = os.path.join(
        configs['datasets_dir'], configs['preprocess_images_dir'])

    os.makedirs(save_path, exist_ok=True)

    for index, row in enumerate(images_list):
        image_gray = resize_image(row, min_height, min_width)
        write_any_image(save_path, row, image_gray)

    print('Modify image count : {}'.format(index+1))
