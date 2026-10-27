import cv2
import argparse
import numpy as np
from numpy import ones, mean
import os
from skimage import io
from PIL import Image
import glob

path = 'F:\cjh\Low latitude\JI91J\\2016'
outpath = 'F:\cjh\Low latitude\JI91J_re\\2016'

os.makedirs(outpath, exist_ok=True)

trainlist = os.listdir(path)
fileId = 0

def cropimage(readpath, savepath):
    # print(savepath)
    print(readpath)
    # 调用cv2.imread读取图片，读入格式为IMREAD_COLOR
    # 使用os.path模块的join方法生成路径
    img_array = cv2.imread(readpath, cv2.IMREAD_UNCHANGED)
    # 调用cv2.resize函数resize图片
    img = img_array[200:531, 177:807, :]
    h, w, c = img.shape
    # 生成图片存储的目标路径
    img[h - 1, :, 0] = 172
    img[h - 1, :, 1] = 177
    img[h - 1, :, 2] = 187

    new_h = h #32
    new_w = w + 39 * 2#39

    newImgData = np.ones((new_h, new_w, c), dtype=np.uint8) * 255

    newImgData[0, :, 0] = 172
    newImgData[0, :, 1] = 177
    newImgData[0, :, 2] = 187

    # newImgData[0 + 20, :, 0] = 172
    # newImgData[0 + 20, :, 1] = 177
    # newImgData[0 + 20, :, 2] = 187
    #
    # newImgData[0 + 20 + 20, :, 0] = 172
    # newImgData[0 + 20 + 20, :, 1] = 177
    # newImgData[0 + 20 + 20, :, 2] = 187
    #
    newImgData[0 : h, 39:new_w-39, :] = img[:, :, :]
    #
    # newImgData[new_h - 1, :, 0] = 172
    # newImgData[new_h - 1, :, 1] = 177
    # newImgData[new_h - 1, :, 2] = 187

    # newImgData[:, :, 0] = 172
    # newImgData[:, :, 1] = 177
    # newImgData[:, :, 2] = 187


    # for i in range(0, 2):
    #     newImgData[:, w + 39 * i - 1, 0] = 172
    #     newImgData[:, w + 39 * i - 1, 1] = 177
    #     newImgData[:, w + 39 * i - 1, 2] = 187

    # 图像第一列灰色对齐，原图像最后一列灰色对齐
    newImgData[:, 0, 0] = 172
    newImgData[:, 0, 1] = 177
    newImgData[:, 0, 2] = 187
    newImgData[:, 39, 0] = 172
    newImgData[:, 39, 1] = 177
    newImgData[:, 39, 2] = 187

    meanOfChanle = np.mean(newImgData, 2)
    meanOfRow = np.mean(meanOfChanle, 1)
    meanOfColumn = np.mean(meanOfChanle, 0)

    # 左上角部分
    # for i in range(0, 32 * 2):
    #     for j in range(0, w):
    #         if meanOfColumn[j] > 160 and meanOfColumn[j] < 205:  # 160 和 205，是超参数，是经验值；
    #             newImgData[i, j, 0] = 172
    #             newImgData[i, j, 1] = 177
    #             newImgData[i, j, 2] = 187

    # 左侧部分
    for i in range(0, h):
        for j in range(0, 39):
            if meanOfRow[i] > 160 and meanOfRow[i] < 205:
                newImgData[i, j, 0] = 172
                newImgData[i, j, 1] = 177
                newImgData[i, j, 2] = 187

    # 右侧部分
    for i in range(0, h):
        for j in range(669, 708):
            if meanOfRow[i] > 160 and meanOfRow[i] < 205:
                newImgData[i, j, 0] = 172
                newImgData[i, j, 1] = 177
                newImgData[i, j, 2] = 187

    # 左下角部分
    # for i in range((32 + 32 + h - 1), new_h):
    #     for j in range(0, w):
    #         if meanOfColumn[j] > 160 and meanOfColumn[j] < 205:
    #             newImgData[i, j, 0] = 172
    #             newImgData[i, j, 1] = 177
    #             newImgData[i, j, 2] = 187

    newImg = cv2.resize(newImgData, (448, 448), interpolation=cv2.INTER_CUBIC)
    save_path = os.path.join(savepath, item)

    # 调用cv2的imwrite函数保存图片
    cv2.imwrite(save_path, newImg)

for item in trainlist:
    m_path = os.path.join(path, item)
    m_num_folder = len(trainlist)  # 获取目标文件夹文件数，对每一个图像文件进行遍历
    print('m:total have %d folders' % (m_num_folder))

    if item.endswith('.png'):
        # 修改图片
        cropimage(m_path, os.path.abspath(outpath))

        # 重命名保存
        # 获取拆分后的名
        #name = os.path.splitext(item)
        # 获取名称后缀
        #ori_name = name[0]
        #picType = name[1]
        src = os.path.join(os.path.abspath(outpath), item)
        #dst = os.path.join(os.path.abspath(outpath),  ori_name + picType)
        #os.rename(src, dst)

        #print('have rename somePic=' + src + ';' + dst)
        fileId += 1
        print('total %d to converted %d png' % (m_num_folder, fileId))