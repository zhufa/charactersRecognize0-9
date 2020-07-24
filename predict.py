# -*- coding: utf-8 -*-
# @Time : 2020/7/24 上午10:41
# @Author : zhufa
# @Software: PyCharm
# @discription: 从带数字的图片中提取数字并进行识别

import tensorflow as tf
import numpy as np
import cv2 as cv


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


img = cv.imread("mydataset/test/0_11.jpg", cv.IMREAD_GRAYSCALE)
_, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
img, cnts, hiera = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnts = sorted(cnts, key=lambda s: cv.boundingRect(s)[0], reverse=False)

x_test = np.array([])
for j in range(len(cnts)):
    [leftTopPiont_x, leftTopPiont_y, width, height] = cv.boundingRect(cnts[j])
    img1 = thresh[leftTopPiont_y:leftTopPiont_y + height, leftTopPiont_x:leftTopPiont_x + width]
    img1 = resizeIMG(img1, 28, interpolation=cv.INTER_LINEAR)
    if x_test.size == 0:
        x_test = img1.reshape((1, 28, 28))
    else:
        x_test = np.concatenate((x_test, img1.reshape((1, 28, 28))))
    # cv.imwrite('mydataset/test/' + str(j)  + '.jpg', img1)

# 加载模型，可以选择模型1还是2
model = tf.keras.models.load_model('./savedmodel/model2/')
# 预览模型结构
model.summary()

# 预测结果1: 整个测试集输入模型去测试
predictions = model.predict(x_test / 255.0)
# tf2.0中，张量的主要属性有三个，分别为形状(shape)、类型(dtype)和值(numpy())，可以通过张量实例的shape、dtype属性和numpy()方法来获取
result = tf.argmax(predictions, 1).numpy()
# 因result中有10000个预测结果，打印会省略
print('predict result is: ' + str(result))
