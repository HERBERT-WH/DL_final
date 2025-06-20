import os
import glob

dir_path = 'D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images'
# for i, file in enumerate(glob.glob(os.path.join(dir_path, '*.jpg'))):
#     os.rename(file, os.path.join(dir_path, f'{i:03d}.jpg'))

import cv2
# import def_Gaussian as dg
# import time
import os.path


# import glob

#####################################################################################################################
# 读取文件夹里面的图像数量 并返回filenum
def countFile(dir):
    # 输入文件夹
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp


filenum = countFile("D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images")  # 返回的是图片的张数
print(filenum)

# filenum
n = 8
index = 1  # 保存图片编号
num = 0  # 处理图片计数
for i in range( filenum ):
    ########################################################
    # 1.读取原始图片
    file_name = f'{i:03d}.jpg'
    if index < 10:
        filename = "D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images/" + file_name
    elif index < 100:
        filename = "D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images/" + file_name
    else:
        filename = "D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images/" + file_name
    print(filename)
    original_image = cv2.imread(filename)
    # 2.下采样
    if n == 4:
        img_1 = cv2.pyrDown(original_image)
        img_1 = cv2.pyrDown(img_1)
    if n == 8:
        img_1 = cv2.pyrDown(original_image)
        img_1 = cv2.pyrDown(img_1)
        img_1 = cv2.pyrDown(img_1)
    # 3.将下采样图片保存到指定路径当中
    if index < 10:
        cv2.imwrite("D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images_8/" + file_name, img_1)
    elif index < 100:
        cv2.imwrite("D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images_8/" + file_name, img_1)
    else:
        cv2.imwrite("D:/Git/nerf-pytorch/data/nerf_llff_data/llfftest/images_8/" + file_name, img_1)

    num = num + 1
    print("正在为第" + str(num) + "图片采样......")
    index = index + 1
