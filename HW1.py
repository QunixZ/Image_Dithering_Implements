import cv2
import numpy as np


img = cv2.imread('cute_dog.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 先转化为灰度图像

size = img.shape  # 读取图像矩阵大小
cv2.namedWindow('Original Image', 1)
cv2.imshow('Original Image', img)
# ########## 1 Binary Image ###############
img_bw = np.zeros(size)
for i in range(0, size[0]):
    for j in range(0, size[1]):
        if img[i, j] < 128:
            img_bw[i, j] = 0
        else:
            img_bw[i, j] = 255
cv2.namedWindow('Binary Image', 1)
cv2.imshow('Binary Image', img_bw)
# ############ 2 Average Dithering ################
re_aver = np.zeros(size)
threshold = 0
for i in range(0, size[0]):
    for j in range(0, size[1]):
        threshold = threshold + img[i, j]/(size[0]*size[1])
# 先利用全图灰度，求出平均灰度，作为二值判定门限值
for i in range(0, size[0]):
    for j in range(0, size[1]):
        if img[i, j] < threshold:
            re_aver[i, j] = 0
        else:
            re_aver[i, j] = 1
cv2.namedWindow('Average Dithering', 1)
cv2.imshow('Average Dithering', re_aver)
# ############ 3 Random Dithering ################
re_rand = np.zeros(size)
for i in range(0, size[0]):
    for j in range(0, size[1]):
        if img[i, j] < np.random.randint(0, 256):  # 生成一个[0,256]区间的随机数，相当于施加随机噪声
            re_rand[i, j] = 0
        else:
            re_rand[i, j] = 1
cv2.namedWindow('Random Dithering', 1)
cv2.imshow('Random Dithering', re_rand)
# ########### 4 Bayer Dithering################
re_bayer = np.zeros(size)
bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])  # 定义一个大小为4*4的bayer矩阵
bayer_matrix = bayer_matrix*16  # 将原256阶灰度级对应映射为16阶灰度级

for i in range(0, size[0]):
    for j in range(0, size[1]):
        x = np.mod(i, 4)
        y = np.mod(j, 4)
        if img[i, j] > bayer_matrix[x, y]:
            re_bayer[i, j] = 255

cv2.namedWindow('Order Dithering', 1)
cv2.imshow('Order Dithering', re_bayer)
# ########### 5 F-S Dithering ################
tmp_fs = np.zeros((size[0]+2, size[1]+2))
tmp_fs[1:size[0]+1, 1:size[1]+1] = img
re_fs = np.zeros(size)

for i in range(1, size[0]+1):  # 直接二值化并求出对应的err，然后按比例进行误差扩散
    for j in range(1, size[1]+1):
        if tmp_fs[i, j] < 128:
            re_fs[i-1, j-1] = 0
            err = tmp_fs[i, j]
        else:
            re_fs[i-1, j-1] = 255
            err = tmp_fs[i, j]-255
        tmp_fs[i, j+1] = tmp_fs[i, j+1]+(7/16)*err
        tmp_fs[i+1, j-1] = tmp_fs[i+1, j-1]+(3/16)*err
        tmp_fs[i+1, j] = tmp_fs[i+1, j]+(5/16)*err
        tmp_fs[i+1, j+1] = tmp_fs[i+1, j+1]+(1/16)*err

cv2.namedWindow('F-S Dithering', 1)
cv2.imshow('F-S Dithering', re_fs)
cv2.waitKey(0)
