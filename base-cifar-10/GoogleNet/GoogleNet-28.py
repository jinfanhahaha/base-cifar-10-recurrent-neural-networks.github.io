
'''

                Base GoogleNet

        conv  :   H    W     channels   size
            conv_1:  224  224    64         7
            pool_1:  112  112    64
            conv_2:  112  112    192        3
            conv_3:  56   56     128        1
            conv_4:  56   56     256        3
            conv_5:  56   56     256        1
            conv_6:  56   56     512        3
            pool_2:  28   28     512
            conv_7: 28   28     256        1
            conv_8: 28   28     512        3
            conv_9: 28   28     256        1
            conv_10: 28   28     512        3
            conv_11: 28   28     256        1
            conv_12: 28   28     512        3
            conv_13: 28   28     256        1
            conv_14: 28   28     512        3
            conv_15: 28   28     256        1
            conv_16: 28   28     1024       3
            pool_3: 14   14     1024
            conv_17: 14   14     512        1
            conv_18: 14   14     1024       3
            conv_19: 14   14     512        1
            conv_20: 14   14     1024       3
            conv_21: 14   14     1024       3
            conv_22: 7    7      1024       3
            conv_23: 7    7      1024       3
            conv_24: 7    7      1024       3
            fc_25:  [500]
            fc_26:  [100]
            fc_27:  [20]
            fc_28:  [10]

            Made by JinFan in 2019.12.01


'''





import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

CIFAR = '../../../DeepLearning/神经网络/神经网络入门/cifar-10'
print(os.listdir(CIFAR))


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# tensorflow.Dataset.
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_val_x, test_val_y = CifarData(test_filenames, True).next_batch(10000)
val_x, val_y = test_val_x[5000:], test_val_y[5000:]
test_x, test_y = test_val_x[:5000], test_val_y[:5000]


def conv_layer(idx, inputs, filters, size, stride):
    '''
            卷积层
            conv_layer:

                    idx:        第idx网络层
                    inputs:     输出网络
                    filters:    输出通道数
                    size:       卷积核大小
                    stride:     步长
        '''
    channels = inputs.get_shape()[3]
    weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[filters]))
    pad_size = size // 2
    pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    inputs_pad = tf.pad(inputs, pad_mat)
    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID', name=str(idx) + "_conv")
    conv_biases = tf.add(conv, biases, name=str(idx) + "_conv_biased")
    print('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
            idx, size, size, stride, filters, int(channels)))

    conv_bn = tf.layers.batch_normalization(conv_biases, beta_initializer=tf.zeros_initializer(),
                                         gamma_initializer=tf.ones_initializer(),
                                         moving_mean_initializer=tf.zeros_initializer(),
                                         moving_variance_initializer=tf.ones_initializer())

    return tf.maximum(0.1 * conv_bn, conv_bn, name=str(idx) + "_leaky_relu")


def pooling_layer(idx, inputs, size, stride):
    '''
            池化层
            pooling_layer:

                    idx:        第idx网络层
                    inputs:     输出网络
                    size:       池化窗口大小
                    stride:     步长
        '''
    print('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride))
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding="SAME",
                          name=str(idx) + "_pool")


def fc_layer(idx, inputs, hiddens, flat=False, linear=False):
    '''
            全连接层
            fc_layer:

                    idx:        第idx网络层
                    inputs:     输出网络
                    hiddens:    输出张量大小
                    flat:       是否进行扁平化输出
                    linear:     是否使用激活函数
        '''
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1] * input_shape[2] * input_shape[3]
        inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
        inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs
    weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
    print('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
            idx, hiddens, int(dim), int(flat), 1 - int(linear)))
    if linear: return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fcb')
    ip = tf.add(tf.matmul(inputs_processed, weight), biases)
    return tf.maximum(0.1 * ip, ip, name=str(idx) + '_fc')  # leaky_relu

with tf.name_scope("build_network"):

    x = tf.placeholder(tf.float32, [None, 3072])
    # [None], eg: [0,5,6,3]
    y = tf.placeholder(tf.int64, [None])

    # [None], eg: [0,5,6,3]
    x_image = tf.reshape(x, [-1, 3, 32, 32])
    # 32*32
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
    lr = tf.Variable(0.001,dtype=tf.float32)
    conv_1 = conv_layer(1, x_image, 64, 7, 2)
    pool_1 = pooling_layer(1, conv_1, 2, 2)
    conv_2 = conv_layer(2, pool_1, 192, 3, 1)
    conv_3 = conv_layer(3, conv_2, 128, 1, 1)
    conv_4 = conv_layer(4, conv_3, 256, 3, 1)
    conv_5 = conv_layer(5, conv_4, 256, 1, 1)
    conv_6 = conv_layer(6, conv_5, 512, 3, 1)
    pool_2 = pooling_layer(2, conv_6, 2, 2)
    conv_7 = conv_layer(7, pool_2, 256, 1, 1)
    conv_8 = conv_layer(8, conv_7, 512, 3, 1)
    conv_9 = conv_layer(9, conv_8, 256, 1, 1)
    conv_10 = conv_layer(10, conv_9, 512, 3, 1)
    conv_11 = conv_layer(11, conv_10, 256, 1, 1)
    conv_12 = conv_layer(12, conv_11, 512, 3, 1)
    conv_13 = conv_layer(13, conv_12, 256, 1, 1)
    conv_14 = conv_layer(14, conv_13, 512, 3, 1)
    conv_15 = conv_layer(15, conv_14, 512, 1, 1)
    conv_16 = conv_layer(16, conv_15, 1024, 3, 1)
    pool_3 = pooling_layer(3, conv_16, 2, 2)
    conv_17 = conv_layer(17, pool_3, 512, 1, 1)
    conv_18 = conv_layer(18, conv_17, 1024, 3, 1)
    conv_19 = conv_layer(19, conv_18, 512, 1, 1)
    conv_20 = conv_layer(20, conv_19, 1024, 3, 1)
    conv_21 = conv_layer(21, conv_20, 1024, 3, 1)
    conv_22 = conv_layer(22, conv_21, 1024, 3, 2)
    conv_23 = conv_layer(23, conv_22, 1024, 3, 1)
    conv_24 = conv_layer(24, conv_23, 1024, 3, 1)
    fc_25 = fc_layer(25, conv_24, 500, flat=True, linear=False)
    fc_26 = fc_layer(26, fc_25, 100, flat=False, linear=False)
    fc_27 = fc_layer(27, fc_26, 20, flat=False, linear=False)
    y_ = fc_layer(28, fc_27, 10, flat=False, linear=True)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    # indices
    predict = tf.argmax(y_, 1)
    # [1,0,1,1,1,0,0,0]
    correct_prediction = tf.equal(predict, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)





with tf.name_scope("train"):
    init = tf.global_variables_initializer()
    batch_size = 30
    train_steps = 2000
    epochs = 10

    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        train_loss_all = []
        for epoch in range(epochs):
            sess.run(tf.assign(lr, 0.0005 * (0.90 ** epoch)))
            train_acc, val_acc = [], []
            for i in range(train_steps):
                batch_data, batch_labels = train_data.next_batch(batch_size)
                loss_train, acc_train, _ = sess.run(
                    [loss, accuracy, train_op],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels})
                train_loss.append(loss_train)
                if (i + 1) % 100 == 0:
                    train_acc.append(acc_train)
                    print('[Train] Epoch: %d Step: %d, loss: %4.5f'
                          % (epoch+1, i + 1, loss_train))
            train_loss_all.append(np.mean(train_loss))
            for i in range(5):
                loss_val, acc_val, _ = sess.run(
                    [loss, accuracy, train_op], feed_dict={
                        x: val_x[i*100:(i+1)*100],
                        y: val_y[i*100:(i+1)*100]
                    })
                val_acc.append(acc_val)
                print('[Val] Epoch: %d Step: %d, loss: %4.5f'
                      % (epoch+1, i + 1, loss_val))
            print("Epoch %d Train_Acc: %4.5f, Val_Acc: %4.5f" %
                (epoch+1, np.mean(train_acc), np.mean(val_acc)))
        loss_test, acc_test, _ = sess.run([loss, accuracy, train_op],
                                          feed_dict={
                                              x: test_x,
                                              y: test_y
                                          })
        print("[Test] Loss %4.5f, Accuracy %4.5f" % (loss_test, acc_test))

        plt.plot(train_loss_all)
        plt.title("Train Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()






