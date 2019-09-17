import tensorflow as tf
import tensorf
import data
import os
import numpy as np

#训练模型里各个参数设置
Batch_size = 64                  #一个训练batch中的图片数, 一个训练 batch 中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
Learning_rate_base = 0.005       # 基础的学习率
Learning_rate_decay = 0.99       # 学习率的衰减率
regularization_rate = 0.00002    # 描述模型复杂度的正则化项在损失函数中的系数
Training_steps = 30000           # 训练轮数
Moving_average_dacay = 0.99      # 滑动平均衰减率
MODEL_SAVE_PATH = "Train_model/"
MODEL_NAME = "photo_model"


def train(train_data, test_data):
    #打印相关模型参数
    print("批大小=",Batch_size, "基础学习率=",Learning_rate_base,"衰减学习率=", Learning_rate_decay,"正则化率=", regularization_rate,"步长衰减=",Moving_average_dacay)
   # 前向传播输入层
    x = tf.placeholder(tf.float32, [Batch_size, tensorf.IMAGE_SIZE, tensorf.IMAGE_SIZE, tensorf.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [Batch_size, tensorf.OUTPUT_NODE], name='y-input')
    to_train = tf.placeholder(tf.int32, [1], name="to-train")
    regularize = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = tensorf.inference(x, to_train, regularize)
    # 存储训练轮数，设置为不可训练
    global_step = tf.Variable(0, trainable=False)
    # 设置滑动平均方法
    variable_averages = tf.train.ExponentialMovingAverage(Moving_average_dacay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 最小化损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)        # 计算每张图片的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)                                                      # 计算当前batch中所有图片的交叉熵平均值
    loss = cross_entropy + tf.add_n(tf.get_collection('losses_regu')) + tf.reduce_sum(tf.image.total_variation(x))   # 总损失等于交叉熵损失和正则化损失的和
    # 设置指数衰减法
    learning_rate = tf.train.exponential_decay(Learning_rate_base,global_step,len(train_data) / Batch_size, Learning_rate_decay, staircase=True)

    #梯度下降
    grad = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(grad, variables_averages_op)
    saver = tf.train.Saver()
#开始训练
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()
        test_set = np.random.permutation(test_data)
        xs_test, ys_test = get_input_and_expect(test_set, 0, Batch_size)
        reshaped_xs_test = np.reshape(xs_test, (Batch_size, 28, 28, 1))
        reshaped_ys_test = np.reshape(ys_test, (Batch_size, 14))
        test_feed = {x: reshaped_xs_test, y_: reshaped_ys_test, to_train: [0]}
#训练次数
        for i in range(Training_steps):
            train_set = np.random.permutation(train_data)
            k = 0
            while True:
                xs, ys = get_input_and_expect(train_set, k, Batch_size)
                if len(xs) < Batch_size:
                    break
                reshaped_xs = np.reshape(xs, (Batch_size, tensorf.IMAGE_SIZE, tensorf.IMAGE_SIZE, tensorf.NUM_CHANNELS))
                reshaped_ys = np.reshape(ys, (Batch_size, tensorf.OUTPUT_NODE))
                train_feed = {x: reshaped_xs, y_: reshaped_ys, to_train: [1]}
                _, losses, step = sess.run([train_op, cross_entropy_mean, global_step], feed_dict=train_feed)
                k += 1
            if i % 1 == 0:
                # 评估模型
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 检验使用滑动平均模型的前向传播的是否正确
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  #计算正确率
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("Epoch is : {0}".format(i), "损失：", losses,' test accuracy is %g ' % test_acc)

def get_input_and_expect(data, index, size):
    data_num = min(len(data) - index * size, size)
    inputs = []
    expects = []
    for j in range(data_num):
        inputs.append(data[index * size + j][0])
        expects.append(data[index * size + j][1])
    return inputs, expects

if __name__ == '__main__':
    tf.app.run()


