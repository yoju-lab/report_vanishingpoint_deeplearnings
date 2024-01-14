import cv2
import numpy as np
import os

# 이미지 경로
img_path = 'datasets/preprocessings/resize_gray__Image_2_074621643184.png'
# img_path = 'datasets/collections/fromBings/20231121122324/buildings/Image_1.jpg'
# 이미지 읽기
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# 이미지 정보 출력
print('Image shape:', img.shape)
print('Pixel values:')
print(' - min:', img.min())
print(' - max:', img.max())
print(' - mean:', img.mean())
print(' - std:', img.std())
print('File size:', os.path.getsize(img_path), 'bytes')
