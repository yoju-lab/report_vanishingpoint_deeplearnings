import os
import cv2
from pytesseract import image_to_string


def move_images_with_classify_text(output_dirs, category, image_path):
    # 이미지 파일 이동
    new_path = os.path.join(
        output_dirs[category], os.path.basename(image_path))

    # 파일명 중복 방지 로직 추가 (동일한 이름의 파일이 존재할 경우 _1, _2 등을 붙임)
    if os.path.isfile(new_path):
        base_name, ext = os.path.splitext(new_path)
        i = 1
        while True:
            new_path = "{}_{:d}{}".format(base_name, i, ext)
            if not os.path.isfile(new_path):
                break
            i += 1

    os.rename(image_path, new_path)


def classify_images(base_dir):
    categories = ['text', 'no_text']
    output_dirs = {category: os.path.join(
        base_dir, category) for category in categories}

    for category in categories:
        if not os.path.exists(output_dirs[category]):
            os.makedirs(output_dirs[category])

    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OCR 적용하여 텍스트 추출
        text = image_to_string(gray_image)
        # 텍스트에서 알파벳과 한글을 추출
        extracted_text = ''.join(c for c in text if c.isalpha())

        # 영어와 한글 포함 여부 확인
        # if any(char.isalpha() for char in text):
        if len(extracted_text) >= 5:
            category = 'text'
        else:
            category = 'no_text'

        puttext = str(len(extracted_text))+','+extracted_text
        cv2.putText(image, puttext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3)

        # 이미지 파일 이동
        move_images_with_classify_text(output_dirs, category, image_path)


if __name__ == '__main__':
    base_dir = 'datasets/'  # 변경 필요: 데이터셋 경로 지정
    classify_images(base_dir)
