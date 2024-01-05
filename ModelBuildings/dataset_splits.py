import os
import shutil
import pandas as pd

# 데이터 세트 경로
DATASET_DIR = "datasets"

# 정보 파일 경로
INFO_FILE_PATH = os.path.join(DATASET_DIR, "any_informations", "01052311_find_vanishingpoints.csv")

# 이미지 경로
IMAGE_DIR = os.path.join(DATASET_DIR, "preprocessings")

# 출력 디렉토리
OUTPUT_DIR_TRAIN = os.path.join(DATASET_DIR, "features", "train")
OUTPUT_DIR_TEST = os.path.join(DATASET_DIR, "features", "test")

# 디렉토리 생성
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

# 정보 파일 읽기
info_df = pd.read_csv(INFO_FILE_PATH)

# 이미지 파일 목록
image_filenames = info_df["file_name"].to_list()

# 이미지 파일 복사
for image_filename in image_filenames:
    shutil.copy(os.path.join(IMAGE_DIR, image_filename), os.path.join(OUTPUT_DIR_TRAIN, image_filename))

# 테스트 세트 분할
test_size = 0.2
num_test_samples = int(len(image_filenames) * test_size)
test_filenames = image_filenames[:num_test_samples]

# 테스트 세트 이미지 파일 복사
for test_filename in test_filenames:
    shutil.copy(os.path.join(IMAGE_DIR, test_filename), os.path.join(OUTPUT_DIR_TEST, test_filename))

# 테스트 및 훈련 세트 좌표 정보 생성
train_info_df = info_df[info_df["file_name"].isin(image_filenames)]
test_info_df = info_df[info_df["file_name"].isin(test_filenames)]

train_info_df.to_csv(os.path.join(DATASET_DIR, "features", "features_train.csv"))
test_info_df.to_csv(os.path.join(DATASET_DIR, "features", "features_test.csv"))
