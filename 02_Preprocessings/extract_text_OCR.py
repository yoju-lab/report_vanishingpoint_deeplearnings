import cv2
from pytesseract import image_to_string
import os


def classify_images(base_dir):
    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:

        # 이미지 로드
        image = cv2.imread(image_path)

        # 그레이스케일 변환
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # 이진화 적용
        _, binary_image = cv2.threshold(
            blurred_image, 127, 255, cv2.THRESH_BINARY)

        # 모폴로지 연산 (침식과 팽창)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_image = cv2.erode(binary_image, kernel)
        dilated_image = cv2.dilate(eroded_image, kernel)

        # 경계 강조 (캐니 에지 검출)
        edges_image = cv2.Canny(dilated_image, threshold1=30, threshold2=100)

        # OCR 적용하여 텍스트 추출
        text = image_to_string(edges_image, lang='kor')

        # 추출된 텍스트에서 한글만 필터링
        extracted_text = ''.join(char for char in text if ord('가')
                                 <= ord(char) <= ord('힣'))

        puttext = str(len(extracted_text))+','+extracted_text

        cv2.putText(image, puttext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3)

        # 영어와 한글 포함 여부 확인
        # if any(char.isalpha() for char in text):
        if len(extracted_text) >= 5:
            category = 'text'
        else:
            category = 'no_text'

        # # 이미지 표시
        cv2.imwrite(image_path, image)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    base_dir = 'datasets/'  # 변경 필요: 데이터셋 경로 지정
    classify_images(base_dir)
