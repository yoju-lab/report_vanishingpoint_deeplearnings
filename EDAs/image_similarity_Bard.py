import cv2
import numpy as np
import os


def calculate_similarity(image1, image2):
    """이미지 유사도를 측정합니다."""

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)

# 숨김 폴더를 제외
def listdir_no_hidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def find_similar_images(images_path):
    """이미지의 유사한 이미지를 찾습니다."""

    # 이미지 파일 목록을 가져옵니다.
    image_paths = listdir_no_hidden(images_path)

    # 이미지 유사도 테이블을 생성합니다.
    similarity_table = {}
    similar_images = []  # Initialize similar_images list

    for image_path1 in image_paths:
        for image_path2 in image_paths:
            if image_path1 == image_path2:
                continue

            similarity = calculate_similarity(
                cv2.imread(os.path.join(images_path, image_path1)),
                cv2.imread(os.path.join(images_path, image_path2)),
            )
            similarity_table[image_path1] = similarity

            if similarity > 0.9:  # Store similar images directly
                similar_images.append(image_path1)

    return similarity_table, similar_images  # Return both values


def delete_similar_images(images_path, similar_images):
    """유사한 이미지를 삭제합니다."""

    for image_path in similar_images:
        os.remove(os.path.join(images_path, image_path))


if __name__ == "__main__":
    images_path = "datasets/preprocessings"
    similarity_table, similar_images = find_similar_images(images_path)  # Receive both values

    # 유사도 결과를 CSV 파일에 저장합니다.
    with open("datasets/any_informations/image_similarity.csv", "w") as f:
        for image_path, similarity in similarity_table.items():
            f.write("{}, {}\n".format(image_path, similarity))

    # 유사한 이미지를 삭제합니다.
    # delete_similar_images(images_path, similar_images)
