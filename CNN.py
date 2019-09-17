import tensorflow as tf
import numpy
from abc import ABCMeta, abstractmethod
import random


class CNN():
    __metaclass__ = ABCMeta

    def __init__(self, rate, input_size, output_size, input_image_construct, weight_construct):
        self.rate = rate
        self._input_data_ = tf.placeholder("float", [None, input_size])
        self._expect_data_ = tf.placeholder("float", [None, output_size])
        self._input_image_ = tf.reshape(self._input_data_, input_image_construct)
        self._weight_, self._bias_, self._convolution_layer_number_, self._reshape_num_ = self._init_para_(
            weight_construct)
        self._keep_prob_ = tf.placeholder("float")
        self._accuracy_ = 0
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _weight_variable_(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable_(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _init_para_(self, weight_construct):
        weight = []
        bias = []
        convolution_layer_number = 0
        reshape_num = 0
        for i in range(len(weight_construct)):
            length = len(weight_construct[i])
            weight.append(self._weight_variable_(weight_construct[i]))
            bias.append(self._bias_variable_([weight_construct[i][length - 1]]))
            if length != 4 and (convolution_layer_number == 0):
                convolution_layer_number = i
                reshape_num = weight_construct[i][0]
        return weight, bias, convolution_layer_number, reshape_num

    def _forward_propagate_(self):
        layer = self._input_image_
        for i in range(self._convolution_layer_number_):
            layer = self.activate(self.convolution(layer, self._weight_[i]), self._bias_[i])
            layer = self.max_pool(layer)

        layer = tf.reshape(layer, [-1, self._reshape_num_])
        length = len(self._weight_)

        for i in range(self._convolution_layer_number_, length - 1):
            layer = self.activate(tf.matmul(layer, self._weight_[i]), self._bias_[i])
            layer = tf.nn.dropout(layer, self._keep_prob_)
        prediction = tf.nn.softmax(tf.matmul(layer, self._weight_[length - 1]) + self._bias_[length - 1])
        return prediction

    def _back_propagate_(self, prediction):
        cross_entropy = -tf.reduce_sum(self._expect_data_ * tf.log(prediction))
        train_step = self.optimizer(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self._expect_data_, 1))
        self._accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return train_step, cross_entropy

    def _get_random_train_data_(self, train_data, batch_size):
        result = random.sample(train_data, batch_size)
        data = []
        label = []
        for i in range(len(result)):
            data.append(result[i][0])
            label.append(result[i][1])
        return numpy.array(data), numpy.array(label)

    def train_random(self, iterations, batch_size, train_data, test_data, test_label):
        prediction = self._forward_propagate_()
        # result = tf.argmax(prediction, 1)
        train_step, error = self._back_propagate_(prediction)
        error_matrix = []
        # file = open('pred.txt', 'w')
        for i in range(iterations):
            batch_xs, batch_ys = self._get_random_train_data_(train_data, batch_size)
            train_step.run(feed_dict={self._input_data_: batch_xs, self._expect_data_: batch_ys, self._keep_prob_: 0.5})
            # error_matrix.append(
            #     error.eval(
            #         feed_dict={self._input_data_: batch_xs, self._expect_data_: batch_ys, self._keep_prob_: 1.0}))
            if i % 100 == 0:
                print("第" + str(i + 1) + "次迭代")
                print(self._accuracy_.eval(feed_dict={
                    self._input_data_: test_data, self._expect_data_: test_label, self._keep_prob_: 1.0}))

        print(self._accuracy_.eval(feed_dict={
            self._input_data_: test_data, self._expect_data_: test_label, self._keep_prob_: 1.0}))
        #         # result = result.eval(feed_dict={
        #         #     self._input_data_: test_data, self._keep_prob_: 1.0})
        #         # for i in range(len(result)):
        #         #     file.write(str(result[i] + 1) + "\n")
        return error_matrix

    def test(self, test_data):
        file = open('pred.txt', 'w')
        prediction = self._forward_propagate_()
        result = tf.argmax(prediction, 1)
        result = result.eval(feed_dict={
            self._input_data_: test_data, self._keep_prob_: 1.0})
        for i in range(len(result)):
            file.write(str(result[i] + 1) + "\n")
        print("test done!")

    @abstractmethod
    def optimizer(self, cross_entropy):
        pass

    @abstractmethod
    def activate(self, mid_result, bias):
        pass

    @abstractmethod
    def convolution(self, x, weight_valve):
        pass

    @abstractmethod
    def max_pool(self, x):
        pass

    def _get_all_train_data_(self, train_data, batch_size):
        numpy.random.shuffle(train_data)
        all_data = []
        all_label = []
        index = 0
        length = len(train_data)
        for i in range(int(length / batch_size)):
            temp_data = []
            temp_label = []
            for j in range(batch_size):
                temp_data.append(train_data[index][0])
                temp_label.append(train_data[index][1])
                index = index + 1
            all_data.append(numpy.array(temp_data))
            all_label.append(numpy.array(temp_label))
        if index < length:
            temp_data = []
            temp_label = []
            for i in range(index, length):
                temp_data.append(train_data[i][0])
                temp_label.append(train_data[index][1])
            all_data.append(numpy.array(temp_data))
            all_label.append(numpy.array(temp_label))
        return all_data, all_label

    def train_all_data(self, epoch, batch_size, train_data, test_data, test_label):
        prediction = self._forward_propagate_()
        train_step, error = self._back_propagate_(prediction)
        error_matrix = []
        train_data, train_label = self._get_all_train_data_(train_data, batch_size)
        for i in range(epoch):
            print("第" + str(i) + "次迭代:")
            for j in range(len(train_data)):
                batch_xs = train_data[j]
                batch_ys = train_label[j]
                train_step.run(
                    feed_dict={self._input_data_: batch_xs, self._expect_data_: batch_ys, self._keep_prob_: 0.5})
                # error_matrix.append(
                #     error.eval(
                #         feed_dict={self._input_data_: batch_xs, self._expect_data_: batch_ys, self._keep_prob_: 1.0}))
                # if j % 100 == 0:
                # print("第" + str(i + 1) + "的第" + str(j) + "次迭代")
            print(self._accuracy_.eval(feed_dict={
                self._input_data_: test_data, self._expect_data_: test_label, self._keep_prob_: 1.0}))

        # print(self._accuracy_.eval(feed_dict={
        #     self._input_data_: test_data, self._expect_data_: test_label, self._keep_prob_: 1.0}))
        return error_matrix
