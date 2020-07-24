# -*- coding: utf-8 -*-
# @Time : 2020/7/24 上午10:39
# @Author : zhufa
# @Software: PyCharm
# @discription: 制作自己的训练数据集
import numpy as np
import cv2 as cv
import os


def resizeIMG(img, size, interpolation):
    h, w = img.shape
    m = max(w, h)
    tmpimg = np.zeros((m, m), dtype=np.uint8)
    start_x = int((m - w) / 2)
    start_y = int((m - h) / 2)
    tmpimg[start_y:start_y + h, start_x:start_x + w] = img
    tmpimg1 = cv.resize(tmpimg, (size - 2, size - 2), interpolation=interpolation)
    resultimg = np.zeros((size, size), dtype=np.uint8)
    resultimg[1:size - 1, 1:size - 1] = tmpimg1
    return resultimg


def splitNumber(fileList, n):
    result = np.array([])
    for i in range(0, n):
        img = cv.imread("mydataset/fontImg0-9/" + str(fileList[i]), cv.IMREAD_GRAYSCALE)
        _, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        img, cnts, hiera = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=lambda s: cv.boundingRect(s)[0], reverse=False)
        for j in range(len(cnts)):
            [leftTopPiont_x, leftTopPiont_y, width, height] = cv.boundingRect(cnts[j])
            img1 = thresh[leftTopPiont_y:leftTopPiont_y + height, leftTopPiont_x:leftTopPiont_x + width]
            img1 = resizeIMG(img1, 28, interpolation=cv.INTER_LINEAR)
            if result.size == 0:
                result = img1.reshape((1, 28, 28))
            else:
                result = np.concatenate((result, img1.reshape((1, 28, 28))))
            cv.imwrite('mydataset/train_img/' + str(j) + '_' + str(i) + '.jpg', img1)
    # 生成数据集过程中对原始数据进行了缩放处理，换种缩放方式就又能再增加一倍数据
    for i in range(10, n + 10):
        img = cv.imread("mydataset/fontImg0-9/" + str(fileList[i - 10]), cv.IMREAD_GRAYSCALE)
        _, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        img, cnts, hiera = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=lambda s: cv.boundingRect(s)[0], reverse=False)
        for j in range(len(cnts)):
            [leftTopPiont_x, leftTopPiont_y, width, height] = cv.boundingRect(cnts[j])
            img1 = thresh[leftTopPiont_y:leftTopPiont_y + height, leftTopPiont_x:leftTopPiont_x + width]
            img1 = resizeIMG(img1, 28, interpolation=cv.INTER_AREA)
            if result.size == 0:
                result = img1.reshape((1, 28, 28))
            else:
                result = np.concatenate((result, img1.reshape((1, 28, 28))))
            cv.imwrite('mydataset/train_img/' + str(j) + '_' + str(i) + '.jpg', img1)
    return result


if __name__ == "__main__":
    fileList = os.listdir("mydataset/fontImg0-9/")
    n = len(fileList)
    result = splitNumber(fileList, n)
    y = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    train_y = y
    for i in range(19):
        train_y = np.concatenate((train_y, y))
    x_test = np.concatenate((result[0:10, :, :], result[190:200, :, :]))
    y_test = np.concatenate((train_y[0:1, :], train_y[19:20, :]))
    np.savez('mydataset/Optical-numeric-characters', x_train=result[10:190, :, :], y_train=train_y[1:19, :].reshape(-1),
             x_test=np.concatenate((result[0:10, :, :], result[190:200, :, :])),
             y_test=np.concatenate((train_y[0:1, :], train_y[19:20, :])).reshape(-1))
