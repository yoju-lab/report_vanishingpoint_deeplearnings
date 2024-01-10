# pandas와 OpenCV 라이브러리를 import 합니다.
import pandas as pd
import cv2

# csv 파일을 pandas의 read_csv() 함수를 사용하여 불러옵니다.
find_vanishingpoints_path = 'datasets/any_informations/01082322_find_vanishingpoints.csv'
df = pd.read_csv(find_vanishingpoints_path)

# 새로 저장할 데이터프레임을 초기화합니다. column은 원본 데이터프레임(df)과 동일합니다.
filtered_df = pd.DataFrame(columns=df.columns)

import numpy as np
import os
# df['filtered']에 대한 exception 처리
try :
    df['filtered']
except :
    df['filtered'] = np.NaN
# df['file_name'].unique()를 통해 중복 없는 file_name을 순회합니다.
unique_rows_df = df[['file_name', 'filtered']].drop_duplicates()
for _, row in unique_rows_df[unique_rows_df['filtered'].isnull()].iterrows():
    # file_name이 같은 row를 가져옵니다.
    same_name_df = df[df['file_name'] == row['file_name']]

    image_list = [None, None]
    # 같은 file_name을 가진 row를 순회합니다.
    for index, row in same_name_df.iterrows():
        # 이미지 파일 경로를 생성합니다. 경로는 'paths' column과 'file_name' column을 합친 것입니다.
        vanishing_point_index = row['vanishing_point_index']
        vanishing_point_filename = '{}_vp{}.png'.format(row['vanishing_point_prefix_filename']
                                               , vanishing_point_index)
        image_path = os.path.join('datasets','find_vanishingpoints',vanishing_point_filename)
        # vanishing_point_index
        # 두 개의 이미지를 불러옵니다.
        image_list[vanishing_point_index-1] = cv2.imread(image_path)

    # 이미지 사이즈가 다를 수 있으므로 두 이미지의 크기를 동일하게 맞추는 작업을 진행합니다.
    # 여기서는 두 이미지의 크기를 더 작은 이미지의 크기에 맞추도록 하겠습니다.
    height = min(image_list[0].shape[0], image_list[1].shape[0])
    width = min(image_list[0].shape[1], image_list[1].shape[1])
    image1 = cv2.resize(image_list[0], (width, height))
    image2 = cv2.resize(image_list[1], (width, height))

    # 이미지에 index 번호를 표시합니다. 
    # cv2.putText 함수를 사용하며, 이 함수는 (이미지, 추가할 텍스트, 텍스트 시작 좌표, 폰트, 폰트 크기, 색상, 두께) 등의 인자를 받습니다.
    cv2.putText(image1, '1', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image2, '2', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # # 두 이미지를 옆으로 붙여서 하나의 이미지로 만듭니다.
    concatenated = np.concatenate((image1, image2), axis=1)

    # # 만들어진 이미지를 화면에 출력합니다.
    cv2.imshow('Images', concatenated)

    # waitKey(0)을 통해 키 입력을 기다립니다.
    key = cv2.waitKey(0) & 0xFF

    # 사용자의 입력에 따라 사용 여부를 판단합니다.
    if key == 2:  # 왼쪽 방향키 in mac, 1번째 이미지 사용
        # 현재 row가 same_name_df의 첫 번째 row인 경우 filtered_df에 추가합니다.
        filtered_df = filtered_df.append(same_name_df.iloc[0])
        df.at[same_name_df.index[0], 'filtered'] = True
        df.at[same_name_df.index[1], 'filtered'] = False
        # 첫 번째 이미지를 선택했으므로 break를 통해 현재 반복문을 종료합니다.
    elif key == 3:  # 오른쪽 방향키 in mac, 2번째 이미지 사용
        # 현재 row가 same_name_df의 두 번째 row인 경우 filtered_df에 추가합니다.
        filtered_df = filtered_df.append(same_name_df.iloc[1])
        df.at[same_name_df.index[0], 'filtered'] = False
        df.at[same_name_df.index[1], 'filtered'] = True
        # 두 번째 이미지를 선택했으므로 break를 통해 현재 반복문을 종료합니다.
    elif key == ord('x'):  # 둘 다 사용 않음
        # 사용하지 않는 이미지이므로 break를 통해 현재 반복문을 종료합니다.
        pass
    elif key == ord('p'):  # 현재까지만 저장 
        # 사용하지 않는 이미지이므로 break를 통해 현재 반복문을 종료합니다.
        df.to_csv(find_vanishingpoints_path)
        break

# 사용여부가 결정된 row를 저장합니다. pandas의 to_csv() 함수를 사용하여 csv 파일로 저장합니다.
if key != ord('p'):
    filtered_df.to_csv('datasets/any_informations/filtering_grayimage_by_vanishing_point.csv'
                    , index=False)
