import os
import time

import cv2
import random
import numpy as np
import shutil
import time
from tqdm import tqdm
from skimage.util import random_noise

random.seed(0)


# 添加噪声函数
def add_noise(image, method, level):
    noisy_image = None
    if method == "gaussian":
        noise = random_noise(image, mode='gaussian', var=level)
        noisy_image = (255 * noise).astype(np.uint8)
    elif method == "poisson":
        noise = random_noise(image, mode='poisson', clip=True, seed=None) * level
        noisy_image = (255 * noise).astype(np.uint8)
    elif method == "salt_pepper":
        noise = random_noise(image, mode='s&p', amount=level)
        noisy_image = (255 * noise).astype(np.uint8)

    return noisy_image


# 参数设置
input_path = r"Z:\ZQY\Hainan\Hainan_classification\HA_no-noise\img_new_label_new"
output_path = r"Z:\ZQY\Hainan\Hainan_classification\image_new_noise_3w"
image_count_threshold = 30000

noise_levels = {
    'gaussian': [0.05, 0.10, 0.15],
    'poisson': [1, 2, 4],
    'salt_pepper': [0.1, 0.15, 0.2]
}

for class_folder in os.listdir(input_path):
    # # 方便调试用的代码
    # if class_folder == '0' or class_folder == '1':
    #     continue
    print()
    print("正在处理 {} 类别图片".format(class_folder))
    folder_path = os.path.join(input_path, class_folder)
    folder_images = os.listdir(folder_path)
    image_num = len(folder_images)

    # 创建输出路径及其子文件夹
    class_folder_output_path = os.path.join(output_path, class_folder)
    if not os.path.exists(class_folder_output_path):
        os.makedirs(class_folder_output_path)

    # 如果原始图片数量超过阈值
    if image_num >= image_count_threshold:
        # 随机筛选20000张图片
        folder_images = random.sample(folder_images, image_count_threshold)

        # 循环保存随机筛选的图片
        for filename in tqdm(folder_images, desc="筛选并保存图片"):
            image_path = os.path.join(folder_path, filename)
            new_image_path = os.path.join(class_folder_output_path, filename)
            shutil.copyfile(image_path, new_image_path)

    else:
        # 循环保存已有的图片
        for filename in tqdm(folder_images, desc="先保存原始图片"):
            image_path = os.path.join(folder_path, filename)
            new_image_path = os.path.join(class_folder_output_path, filename)
            shutil.copyfile(image_path, new_image_path)

        # 随机循环添加噪声
        while image_num < image_count_threshold:
            for filename in folder_images:
                # 生成随机变量
                random_index = random.randint(0, 8)

                # 根据随机变量确定添加噪声方法
                if random_index < 3:
                    method = "gaussian"
                    level = noise_levels["gaussian"][random_index]
                elif 3 <= random_index < 6:
                    method = "poisson"
                    level = noise_levels["poisson"][random_index - 3]
                else:
                    method = "salt_pepper"
                    level = noise_levels["salt_pepper"][random_index - 6]

                print("\r加噪并保存图片: {}/{}".format(image_num, image_count_threshold), end="")
                # 这行判定同样不能删除
                if image_num >= image_count_threshold:
                    break

                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                # 确定添加列和块噪声的数目，均是范围内的随机数
                col_num = random.randint(0, 3)
                rec_num = random.randint(1, 3) if col_num == 0 else random.randint(0, 3)

                h, w, _ = image.shape

                # 第一步：选择图片image的某段连续的1-5列添加噪声
                for i in range(col_num):
                    start_col = random.randint(0, w-5)
                    end_col = start_col + random.randint(1, 5)
                    image[:, start_col:end_col] = add_noise(image[:, start_col:end_col], method, level)


                # 第二步：对图片image的某个a*b的平行四边形区域添加噪声，平行四边形区域为45度角斜向右上角
                rec_b = random.randint(35, 45)  # 底
                rec_h = random.randint(18, 25)  # 高

                for i in range(rec_num):
                    start_col = random.randint(0, w - rec_h)
                    start_row = random.randint(rec_h - 1, h - rec_b - rec_h)  # 最小可以取rec_h - 1；减去平行四边形底、延申部分
                    for j in range(rec_h):
                        noise_area = image[start_row:start_row + rec_b + rec_h, start_col:start_col+1]
                        image[start_row:start_row+rec_b+rec_h, start_col:start_col+1] = add_noise(noise_area, method,
                                                                                                  level)
                        start_col += 1
                        start_row -= 1

                new_filename = '{}_{}_{}_{}.png'.format(filename.split('.')[0], method, level, image_num+1)  # 防止名称重复！
                new_image_path = os.path.join(class_folder_output_path, new_filename)
                cv2.imwrite(new_image_path, image)

                image_num += 1

    time.sleep(0.5)

print("\n处理完成")
