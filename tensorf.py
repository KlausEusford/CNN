import tensorflow as tf
#输入图片属性像素格式
INPUT_NODE = 784
OUTPUT_NODE = 14
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 14

#第1，2，3层卷积层
CONV1_DEEP = 6
CONV1_SIZE = 5

CONV2_DEEP = 16
CONV2_SIZE = 5

CONV3_DEEP = 16
CONV3_SIZE = 5

FC_SIZE1 = 120
FC_SIZE2 = 84

STDDEV = 0.1
MAXVAL = 0.1
MINVAL = -0.1

def inference(input_tensor, train, regularize):
    #第一层权重，偏置，输出，激活函数参数
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.random_uniform_initializer(minval=MINVAL, maxval=MAXVAL))
        conv1_out = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1_out, conv1_biases))

#最大池化层maxpooling
    with tf.variable_scope("layer2-pool1"):
        pool1_out = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#第二层卷积层参数
    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV1_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.random_uniform_initializer(minval=MINVAL, maxval=MAXVAL))
        conv3_out = tf.nn.conv2d(pool1_out, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3_relu = tf.nn.relu(tf.nn.bias_add(conv3_out, conv3_biases))
#第二层的maxpooling
    with tf.variable_scope("layer6-pool3"):
        pool3_out = tf.nn.max_pool(conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool3_shape = pool3_out.get_shape().as_list()
        nodes_num = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]
        pool3_reshaped = tf.reshape(pool3_out, [pool3_shape[0], nodes_num])
#全连接1
    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes_num, FC_SIZE1],
                                      initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularize is not None:
            tf.add_to_collection('losses_regu', regularize(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE1], initializer=tf.random_uniform_initializer(minval=MINVAL, maxval=MAXVAL))

        fc1_out = tf.nn.sigmoid(tf.matmul(pool3_reshaped, fc1_weights) + fc1_biases)
        if train[0] == 1:
            fc1_out = tf.nn.dropout(fc1_out, 0.5)
#全连接2
    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE1, FC_SIZE2],
                                      initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularize is not None:
            tf.add_to_collection('losses_regu', regularize(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FC_SIZE2], initializer=tf.random_uniform_initializer(minval=MINVAL, maxval=MAXVAL))
        fc2_out = tf.nn.sigmoid(tf.matmul(fc1_out, fc2_weights) + fc2_biases)
        if train[0] == 1:
            fc2_out = tf.nn.dropout(fc2_out, 0.5)
#最后输出，并计算之前的损失函数
    with tf.variable_scope('layer8-out'):
        out_weights = tf.get_variable("weight", [FC_SIZE2, NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularize is not None:
            tf.add_to_collection('losses_regu', regularize(out_weights))
        out_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.random_uniform_initializer(minval=MINVAL, maxval=MAXVAL))
        result = tf.matmul(fc2_out, out_weights) + out_biases

    return result
