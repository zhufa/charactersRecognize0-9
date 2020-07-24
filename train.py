# -*- coding: utf-8 -*-
# @Time : 2020/7/24 上午10:35
# @Author : zhufa
# @Software: PyCharm
# @discription: 识别0-9印刷数字

import tensorflow as tf
import numpy as np

'''
官方代码使用mnist.load_data()加载数据集，但需要联网下载，但国内可能下载不了
解决办法：自己从网上下载数据集，复制mnist.load_data()里的加载数据集的代码，自己写方法实现

加载数据集官方代码：
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
'''


# 方法内的代码复制自mnist.load_data()
def loadMinstData(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


path = 'mydataset/Optical-numeric-characters.npz'
(x_train, y_train), (x_test, y_test) = loadMinstData(path)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 随机打乱训练集
np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)

# Keras 中有两类主要的模型：Sequential 顺序模型 和 使用函数式 API 的 Model 类模型，本案例采用前者

# 模型结构搭建：使用tensorflow官方minist手写数字的模型
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name='model1')

# 模型结构搭建：使用tensorflow1.0时代官方minist手写数字的模型
model2 = tf.keras.models.Sequential([
    # 数据集里为60000张28×28的测试图片，Conv2D()需要4维的张量：(batch_size, rows, cols, channels)，默认channels在后面，可更改
    # batch_size会在训练的时候自动加上，所以需要将原28×28的input转换为 28×28×1 的input，指明channels通道数为1
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # 作为第一层，需要指明input_shape参数值
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu',
                           bias_initializer=tf.keras.initializers.constant(0.1)),
    tf.keras.layers.MaxPool2D(strides=[2, 2], padding='same'),
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                           bias_initializer=tf.keras.initializers.constant(0.1)),
    tf.keras.layers.MaxPool2D(strides=[2, 2], padding='SAME'),
    tf.keras.layers.Reshape([7 * 7 * 64]),
    tf.keras.layers.Dense(1024, activation='relu', bias_initializer=tf.keras.initializers.constant(0.1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax', bias_initializer=tf.keras.initializers.constant(0.1))
], name='model2')

# 选择模型
model = model2
# 查看模型结构
model.summary()
# 配置模型如损失函数和优化器等参数，准备训练
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，batch_size默认32，那么每个batch(每批)训练32张图片，每跑完一次epoch(跑完整个训练集)需要迭代iterations(step)=60000/32=1875次
model.fit(x_train, y_train, batch_size=25, epochs=15)
# 测试模型的损失值和指标值（此处指精度，因为metrics=['accuracy']）
model.evaluate(x_test, y_test)

# 查看模型预测的结果
predictions = model.predict(x_test)
result = tf.argmax(predictions, 1).numpy()
print('the x_test img predict result is: ' + str(result))  # 结果太多时输出会省略

# 两个模型分别保存
model.save('./savedmodel/' + model.name + '/')
