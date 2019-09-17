from PIL import Image
import numpy as np

#读取图片信息的接口

def read_photo():
    path_head = "TRAIN/"
    image = []
    for i in range(1, 15):
        path_package = path_head + str(i) + "/"
        image_package = []
        for j in range(256):
            path = path_package + str(j) + ".bmp"
            im = np.array(Image.open(path))
            im = im.reshape((1, 784))
            image_package.append(im)
        image.append(image_package)
    return image

#读取测试集图片信息
def read_photo_test(path):
    im = np.array(Image.open(path))
    im = im.reshape((1, 784))
    return im[0]

#把图片信息分部存入对于矩阵里备用
def get_train_and_test_data():
    input_sets = read_photo()
    types = len(input_sets)
    train_sets = []
    test_sets = []

    for i in range(types):
        output = np.zeros(types)
        output[i] = 1
        data_num = len(input_sets[i])
        train_num = int(data_num / 16 * 15)
        test_num = data_num - train_num
        for j in range(train_num):
            train_sets.append([input_sets[i][j][0], output])
        for k in range(1, test_num):
            test_sets.append([input_sets[i][-k][0], output])
    return np.array(train_sets), np.array(test_sets)
