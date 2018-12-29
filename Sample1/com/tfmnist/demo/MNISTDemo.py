# -*- coding: utf-8 -*-
# TensorFlow and tf.keras
from com.tfmnist.demo.utils import mnist_reader
from tensorflow import keras
import tensorflow as tf
# Helper libraries
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
       matplotlib   是一个 Python 的 2D绘图库
       image 和标签数据集
   '''
    train_images, train_labels = mnist_reader.load_mnist('data', kind='train')
    test_images, test_labels = mnist_reader.load_mnist('data', kind='t10k')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("train_images ", train_images.shape, " test_images", test_images.shape)
    '''
    预处理数据
    使用2D绘图库来显示第一张 image图片
    '''
    plt.figure()
    # image_temp=train_images[0].reshape(28, 28)
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)

    train_images = train_images / 255.0
    test_images = train_images / 255.0

    '''
    显示训练集中的前 25 张图像，并在每张图像下显示类别名称
    '''
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])

    '''
    构建模型
    '''

