# -*- coding: utf-8 -*-
# TensorFlow and tf.keras
from com.tfmnist.demo.utils import mnist_reader
from tensorflow import keras
import tensorflow as tf
# Helper libraries
import numpy as np

import matplotlib.pyplot as plt


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


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
    print(train_images[0])
    '''
    预处理数据
    使用2D绘图库来显示第一张 image图片
    '''
    # plt.figure()
    # image_temp=train_images[0].reshape(28, 28)
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)

    '''
    预处理数据
    '''
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    '''
    显示训练集中的前 25 张图像，并在每张图像下显示类别名称
    '''
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])

    '''
    构建模型
    '''

    '''
    设置层
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    '''
    编译模型
    '''
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    '''
      开始训练        epochs训练次数
    '''
    model.fit(train_images, train_labels, epochs=5)

    '''
    评估准确率  比较一下模型在测试数据集上的表现
    evaluate 评估
    '''
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    '''
      做出预测
    '''
    predictions = model.predict(test_images)

    # index = 4000
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image(index, predictions, test_labels, test_images)
    # plt.subplot(1, 2, 2)
    # plot_value_array(index, predictions, test_labels)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
