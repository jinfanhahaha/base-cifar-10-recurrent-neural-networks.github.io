
'''
            VGG-13
        conv1 3x1 1
        conv2 1x3 1
        conv3 3x1 1
        conv4 1x3 1
        pool1 2x2 2
        conv5 1x3 1
        conv6 3x1 1
        conv7 1x3 1
        conv8 3x1 1
        pool2 2x2 2
        conv9  1x3 1
        conv10 3x1 1
        conv11 1x3 1
        conv12 3x1 1
        pool3 2x2 2
        fc13 -> 10 + softmax

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

def conv(x, output_channel, kernel_size, name):
    return tf.layers.conv2d(x,
                            output_channel,
                            kernel_size=kernel_size,
                            padding="same",
                            activation=tf.nn.relu,
                            name=name
                           )

with tf.name_scope("build_network"):
    x = tf.placeholder(tf.float32, [None, 3072])
    # [None], eg: [0,5,6,3]
    y = tf.placeholder(tf.int64, [None])

    # [None], eg: [0,5,6,3]
    x_image = tf.reshape(x, [-1, 3, 32, 32])
    # 32*32
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

    # conv1: 神经元图， feature_map, 输出图像
    conv1 = conv(x_image, 32, (3, 1), "conv1")
    conv2 = conv(conv1, 32, (1, 3), "conv2")
    conv3 = conv(conv2, 32, (3, 1), "conv3")
    conv4 = conv(conv3, 32, (1, 3), "conv4")

    # 16 * 16
    pooling1 = tf.layers.max_pooling2d(conv4,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool1')

    conv5 = conv(pooling1, 32, (1, 3), "conv5")
    conv6 = conv(conv5, 32, (3, 1), "conv6")
    conv7 = conv(conv6, 32, (1, 3), "conv7")
    conv8 = conv(conv7, 32, (3, 1), "conv8")

    # 8 * 8
    pooling2 = tf.layers.max_pooling2d(conv8,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool2')

    conv9 = conv(pooling2, 32, (1, 3), "conv9")
    conv10 = conv(conv9, 32, (3, 1), "conv10")
    conv11 = conv(conv10, 32, (1, 3), "conv11")
    conv12 = conv(conv11, 32, (3, 1), "conv12")

    # 4 * 4 * 32
    pooling3 = tf.layers.max_pooling2d(conv12,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool3')
    # [None, 4 * 4 * 32]
    flatten = tf.layers.flatten(pooling3)
    y_ = tf.layers.dense(flatten, 10)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    # indices
    predict = tf.argmax(y_, 1)
    # [1,0,1,1,1,0,0,0]
    correct_prediction = tf.equal(predict, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)





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






