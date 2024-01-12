# pandas와 OpenCV 라이브러리를 import 합니다.
import pandas as pd

# csv 파일을 pandas의 read_csv() 함수를 사용하여 불러옵니다.
find_vanishingpoints_path = 'datasets/any_informations/01122202_find_vanishingpoints.csv'
df = pd.read_csv(find_vanishingpoints_path)

# 짝수 행에는 True, 홀수 행에는 False를 입력하는 filtered column 생성
df['filtered'] = df.index % 2 == 0

df.to_csv('datasets/any_informations/01122202_filtering_grayimage_by_vanishing_point.csv'
                    , index=False)
