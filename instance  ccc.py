import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import random
from PIL import Image
from CNN import CNN


def get_data():
    train_data = []

    for i in range(14):
        image_label = numpy.zeros(14)
        image_label[i] = 1
        for j in range(200):
            image = Image.open('./TRAIN/' + str(i + 1) + '/' + str(j) + '.bmp')
            image_array = numpy.array(image)
            image_matrix = []
            for a in range(len(image_array)):
                for b in range(len(image_array[a])):
                    image_matrix.append(image_array[a][b])
            train_data.append([image_matrix, image_label])

    test_data = []
    test_label = []
    for i in range(14):
        image_label = numpy.zeros(14)
        image_label[i] = 1
        for j in range(56):
            image = Image.open('./TRAIN/' + str(i + 1) + '/' + str(200 + j) + '.bmp')
            image_array = numpy.array(image)
            image_matrix = []
            for a in range(len(image_array)):
                for b in range(len(image_array[a])):
                    image_matrix.append(image_array[a][b])
            test_data.append(image_matrix)
            test_label.append(image_label)

    # 测试代码
    # for i in range(763):
    #     image = Image.open('./test/' + str(i) + '.bmp')
    #     image_array = numpy.array(image)
    #     image_matrix = []
    #     for a in range(len(image_array)):
    #         for b in range(len(image_array[a])):
    #             image_matrix.append(image_array[a][b])
    #     test_data.append(image_matrix)

    return train_data, numpy.array(test_data), numpy.array(test_label)


class Instance(CNN):
    def optimizer(self, cross_entropy):
        optimizer = tf.train.GradientDescentOptimizer(self.rate).minimize(cross_entropy)
        # optimizer = tf.train.AdamOptimizer(self.rate).minimize(cross_entropy)
        # self.sess.run(tf.global_variables_initializer())
        return optimizer

    def activate(self, mid_result, bias):
        return tf.nn.relu(mid_result + bias)

    def convolution(self, x, weight_valve):
        return tf.nn.conv2d(x, weight_valve, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


train_data, test_data, test_label = get_data()
cnn_instance_1 = Instance(0.0001, 784, 14, [-1, 28, 28, 1],
                          [[4, 4, 1, 32], [4, 4, 32, 64], [7 * 7 * 64, 1024], [1024, 14]])

# cnn_instance_2 = Instance(0.0001, 784, 14, [-1, 28, 28, 1],
#                           [[4, 4, 1, 32], [4, 4, 32, 64], [7 * 7 * 64, 1024], [1024, 14]])

error_matrix = cnn_instance_1.train_random(1500, 150, train_data, test_data, test_label)

print("input your test file path:")
test_path = input()
print("test start:")
test_data = []
for i in range(763):
    image = Image.open(test_path + str(i) + '.bmp')
    image_array = numpy.array(image)
    image_matrix = []
    for a in range(len(image_array)):
        for b in range(len(image_array[a])):
            image_matrix.append(image_array[a][b])
    test_data.append(image_matrix)
cnn_instance_1.test(test_data)
# cnn_instance_1.train_all_data(100, 50, train_data, test_data, test_label)
# cnn_instance_2.train_random(2000, 150, train_data, test_data, test_label)
# plt.title("batch_size= 50,rate=0.0001 , iterations=2000")
# plt.plot(error_matrix, color='r')
# plt.show()
